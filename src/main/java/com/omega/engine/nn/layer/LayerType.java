package com.omega.engine.nn.layer;

/**
 * layer type
 *
 * @author Administrator
 */
public enum LayerType {
    full("full"), softmax("softmax"), conv("conv"), conv_transpose("conv_transpose"), cbl("cbl"), double_conv("double_conv"), unet_up("unet_up"), unet_mid("unet_mid"), unet_down("unet_down"), pooling("pooling"), input("input"), softmax_cross_entropy("softmax_cross_entropy"), sigmod("sigmod"), swish("swish"), relu("relu"), gelu("gelu"), leakyRelu("leakyRelu"), tanh("tanh"), silu("silu"), bn("bn"), layer_norm("layer_norm"), group_norm("group_norm"), rms_norm("rms_norm"), instance_normal("instance_normal"), block("block"), shortcut("shortcut"), avgpooling("avgpooling"), upsample("upsample"), duffsion_upsample("duffsion_upsample"), duffsion_res_block("duffsion_res_block"), route("route"), yolo("yolo"), rnn("rnn"), mutli_head_attention("mutli_head_attention"), poswise_feed_forward("poswise_feed_forward"), mlp("mlp"), clip_mlp("clip_mlp"), bert_inter("bert_inter"), bert_output("bert_output"), bert_encoder("bert_encoder"), feed_forward("feed_forward"), transformer_encoder("transformer_encoder"), transformer_decoder("transformer_decoder"), transformer_block("transformer_block"), lstm("lstm"), embedding("embedding"), rope("rope"), dropout("dropout"), clip_vision_embedding("clip_vision_embedding"), time_embedding("time_embedding"), lpips("lpips");
    private String key;

    LayerType(String key) {
        this.key = key;
    }

    public static LayerType getEnumByKey(String key) {
        for (LayerType type : LayerType.values()) {
            if (type.getKey().equals(key)) {
                return type;
            }
        }
        return null;
    }

    public String getKey() {
        return key;
    }
}

