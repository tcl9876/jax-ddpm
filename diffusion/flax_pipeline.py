import warnings
from functools import partial
from typing import Dict, List, Optional, Union, Any

import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

from dpm import Model as DiffusionWrapper
from diffusers.models.vae_flax import FlaxAutoencoderKL
from diffusers.schedulers import FlaxDDIMScheduler, FlaxLMSDiscreteScheduler, FlaxPNDMScheduler
from diffusers.utils import logging
from schedules import get_logsnr_schedule
#from . import FlaxStableDiffusionPipelineOutput
#from .safety_checker_flax import FlaxStableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FlaxGeneralDiffusionPipeline: #when inheriting from FlaxDiffusionPipeline, there's an error when registering modules
    r"""
    A general purpose pipeline to generate samples. Supports pixel-based DDPMs and Latent DDPMs, either text-to-image or class-conditional/unconditional.
    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`FlaxCLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.FlaxCLIPTextModel),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`FlaxUNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], or [`FlaxPNDMScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: Optional[FlaxAutoencoderKL],
        text_encoder: Optional[FlaxCLIPTextModel],
        tokenizer: Optional[CLIPTokenizer],
        unet: Any, #FlaxUNet2DConditionModel,
        scheduler: Union[FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler],
        safety_checker: None, #Optional[FlaxStableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPFeatureExtractor],
        dtype: jnp.dtype = jnp.float32,
        model_config: Optional[Any] = None, #put after for now
    ):
        super().__init__()
        self.dtype = dtype

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.model_config = model_config
        assert model_config is not None
    
    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _get_has_nsfw_concepts(self, features, params):
        if self.safety_checker is None:
            return False

        has_nsfw_concepts = self.safety_checker(features, params)
        return has_nsfw_concepts

    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # safety_model_params should already be replicated when jit is True
        pil_images = [Image.fromarray(image) for image in images]
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        if jit:
            features = shard(features)
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            safety_model_params = unreplicate(safety_model_params)
        else:
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts

    def _generate(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: Optional[jax.random.PRNGKey],
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        latents: Optional[jnp.array] = None,
        debug: bool = False,
    ):
        batch_size = len(prompt_ids) #prompt_ids.shape[0]
        if self.vae is not None:
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        else:
            latents_shape = (batch_size, 3, height, width)

        if self.text_encoder is not None: #allow None text encoder for class-conditional or unconditional
            # get prompt text embeddings
            text_embeddings = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

            max_length = prompt_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids, params=params["text_encoder"])[0]
            context = jnp.concatenate([uncond_embeddings, text_embeddings])
        else:
            context = None

        if latents is None:
            latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        model_fn = lambda x, logsnr: self.unet.apply(
			{'params': params["unet"]}, x=x, logsnr=logsnr, y=None, train=False) #TODO: allow for label and text input.
        
        margs = self.model_config
        model_wrap = DiffusionWrapper(
			model_fn=model_fn,
			mean_type=margs.mean_type,
			logvar_type=margs.logvar_type,
			logvar_coeff=margs.get('logvar_coeff', 0.)
        )
        logsnr_schedule_fn=get_logsnr_schedule(
				**margs.eval_logsnr_schedule)
        
        latents = jnp.transpose(latents, [0, 2, 3, 1]) #prog-dist unet takes in NHWC, may change if we switch to diffusers Unet
        def loop_body(step, args):
            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if guidance_scale > 1.0:
                latents_input = jnp.concatenate([latents] * 2)
            else:
                latents_input = latents 

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0]).astype(latents_input.dtype)
            latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)
            logsnr_t = logsnr_schedule_fn((timestep + 1) / (scheduler_state.timesteps.max() + 1))

            noise_pred = model_wrap._run_model(
                z=jnp.array(latents_input),
                logsnr=logsnr_t, #jnp.full((latents.shape[0],), logsnr_t),
                model_fn=model_fn,
                clip_x=True
            )["model_eps"]

            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state #DO jnp.where(i == 0, x_pred_t, z_s_pred), this fixes adding noise of beta_1 at the end.

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if debug:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        if self.vae is not None: #allow VAE to be None for pixel space ddpm
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample
        else:
            image = latents
            
        image = (image / 2 + 0.5).clip(0, 1)#.transpose(0, 2, 3, 1) #prog-dist unet takes in NHWC, may add back if we switch to diffusers Unet
        return image

    def __call__(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: Optional[jax.random.PRNGKey],
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        latents: jnp.array = None,
        return_dict: bool = True,
        jit: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        if jit:
            images = _p_generate(
                self, prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, debug
            )
        else:
            images = self._generate(
                prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, debug
            )

        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            images_uint8_casted = (images * 255).round().astype("uint8")
            num_devices, batch_size = images.shape[:2]

            images_uint8_casted = np.asarray(images_uint8_casted).reshape(num_devices * batch_size, height, width, 3)
            images_uint8_casted, has_nsfw_concept = self._run_safety_checker(images_uint8_casted, safety_params, jit)
            images = np.asarray(images)

            # block images
            if any(has_nsfw_concept):
                for i, is_nsfw in enumerate(has_nsfw_concept):
                    if is_nsfw:
                        images[i] = np.asarray(images_uint8_casted[i])

            images = images.reshape(num_devices, batch_size, height, width, 3)
        else:
            has_nsfw_concept = False

        if not return_dict:
            return (images, has_nsfw_concept)

        return images #FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


# TODO: maybe use a config dict instead of so many static argnums
@partial(jax.pmap, static_broadcasted_argnums=(0, 4, 5, 6, 7, 9))
def _p_generate(
    pipe, prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, debug
):
    return pipe._generate(
        prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, debug
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))
def _p_get_has_nsfw_concepts(pipe, features, params):
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)