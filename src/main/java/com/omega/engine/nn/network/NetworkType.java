package com.omega.engine.nn.network;

/**
 * 
 * @author Administrator
 *
 */
public enum NetworkType {
	
	BP("BP"),
	CNN("CNN"),
	ANN("ANN"),
	RNN("RNN"),
	SEQ2SEQ_RNN("SEQ2SEQ_RNN"),
	SEQ2SEQ("SEQ2SEQ"),
	TTANSFORMER("TTANSFORMER"),
	GPT("GPT"),
	LLAMA("LLAMA"),
	LLAMA2("LLAMA2"),
	LLAMA3("LLAMA3"),
	LLAVA("LLAVA"),
	VAE("VAE"),
	VQVAE("VQVAE"),
	CLIP_VISION("CLIP_VISION"),
	CLIP_TEXT("CLIP_TEXT"),
	UNET("UNET"),
	DUFFSION_UNET("DUFFSION_UNET"),
	DUFFSION_UNET_COND("DUFFSION_UNET_COND"),
	PATCH_GAN_DISC("PATCH_GAN_DISC"),
	YOLO("YOLO");
	
	NetworkType(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}

}
