package com.omega.engine.nn.layer.diffusion.unet;

import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffusion.TimeEmbeddingLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

/**
 * UNetCond2
 *
 * @author Administrator
 */
public class UNetCond2 extends Layer {
    public TimeEmbeddingLayer t_embd;
    public ConvolutionLayer conv_in;
    public List<UNetDownBlock> downs;
    public UNetResidualBlock down_res;
    public UNetMidBlock mids;
    public UNetResidualBlock up_res;
    public List<UNetUpBlock> ups;
    public ConvolutionLayer conv_out;
    public GNLayer norm;
    public ConvolutionLayer conv_final;
    private int timeSteps;
    private int nTime;
    private int contextTime;
    private int contextDim;
    private int numLayer;
    private int headNum;
    private int groupNum = 32;
    private int[] downChannels;
    private RouteLayer cat_res;
    private List<RouteLayer> cats;
    private RouteLayer cat_out;
    private SiLULayer act;
    private Tensor tDiff;

    public UNetCond2(int channel, int height, int width, int[] downChannels, int headNum, int nTime, int timeSteps, int contextTime, int contextDim, int numLayer, int groupNum, Network network) {
        this.network = network;
        this.nTime = nTime;
        this.groupNum = groupNum;
        this.channel = channel;
        this.oChannel = channel;
        this.height = height;
        this.width = width;
        this.oHeight = height;
        this.oWidth = width;
        this.downChannels = downChannels;
        this.headNum = headNum;
        this.nTime = nTime;
        this.timeSteps = timeSteps;
        this.contextTime = contextTime;
        this.contextDim = contextDim;
        this.numLayer = numLayer;
        this.groupNum = groupNum;
        initLayers();
    }

    public static void loadWeight(Map<String, Object> weightMap, UNetCond2 network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * t_proj

         */
        ClipModelUtils.loadData(network.t_embd.linear1.weight, weightMap, "time_embedding.linear_1.weight");
        ClipModelUtils.loadData(network.t_embd.linear1.bias, weightMap, "time_embedding.linear_1.bias");
        ClipModelUtils.loadData(network.t_embd.linear2.weight, weightMap, "time_embedding.linear_2.weight");
        ClipModelUtils.loadData(network.t_embd.linear2.bias, weightMap, "time_embedding.linear_2.bias");
        /**
         * conv_in

         */
        ClipModelUtils.loadData(network.conv_in.weight, weightMap, "unet.encoders.0.0.weight");
        ClipModelUtils.loadData(network.conv_in.bias, weightMap, "unet.encoders.0.0.bias");
        for (int i = 0; i < 2; i++) {
            int idx = i * 2 + 1;
            /**
             * resnet block

             */
            network.downs.get(i).res[0].gn_feature.gamma = ClipModelUtils.loadData(network.downs.get(i).res[0].gn_feature.gamma, weightMap, 1, "unet.encoders." + idx + ".0.groupnorm_feature.weight");
            network.downs.get(i).res[0].gn_feature.beta = ClipModelUtils.loadData(network.downs.get(i).res[0].gn_feature.beta, weightMap, 1, "unet.encoders." + idx + ".0.groupnorm_feature.bias");
            ClipModelUtils.loadData(network.downs.get(i).res[0].conv_feature.weight, weightMap, "unet.encoders." + idx + ".0.conv_feature.weight");
            ClipModelUtils.loadData(network.downs.get(i).res[0].conv_feature.bias, weightMap, "unet.encoders." + idx + ".0.conv_feature.bias");
            ClipModelUtils.loadData(network.downs.get(i).res[0].temb.linear.weight, weightMap, "unet.encoders." + idx + ".0.linear_time.weight");
            ClipModelUtils.loadData(network.downs.get(i).res[0].temb.linear.bias, weightMap, "unet.encoders." + idx + ".0.linear_time.bias");
            network.downs.get(i).res[0].gn_merged.gamma = ClipModelUtils.loadData(network.downs.get(i).res[0].gn_merged.gamma, weightMap, 1, "unet.encoders." + idx + ".0.groupnorm_merged.weight");
            network.downs.get(i).res[0].gn_merged.beta = ClipModelUtils.loadData(network.downs.get(i).res[0].gn_merged.beta, weightMap, 1, "unet.encoders." + idx + ".0.groupnorm_merged.bias");
            ClipModelUtils.loadData(network.downs.get(i).res[0].conv_merged.weight, weightMap, "unet.encoders." + idx + ".0.conv_merged.weight");
            ClipModelUtils.loadData(network.downs.get(i).res[0].conv_merged.bias, weightMap, "unet.encoders." + idx + ".0.conv_merged.bias");
            network.downs.get(i).res[0].residual_layer.weight = ClipModelUtils.loadData(network.downs.get(i).res[0].residual_layer.weight, weightMap, 4, "unet.encoders." + idx + ".0.residual_layer.weight");
            ClipModelUtils.loadData(network.downs.get(i).res[0].residual_layer.bias, weightMap, "unet.encoders." + idx + ".0.residual_layer.bias");
            /**
             * attn block

             */
            network.downs.get(i).attns[0].gn.gamma = ClipModelUtils.loadData(network.downs.get(i).attns[0].gn.gamma, weightMap, 1, "unet.encoders." + idx + ".1.groupnorm.weight");
            network.downs.get(i).attns[0].gn.beta = ClipModelUtils.loadData(network.downs.get(i).attns[0].gn.beta, weightMap, 1, "unet.encoders." + idx + ".1.groupnorm.bias");
            network.downs.get(i).attns[0].conv_in.weight = ClipModelUtils.loadData(network.downs.get(i).attns[0].conv_in.weight, weightMap, 4, "unet.encoders." + idx + ".1.conv_input.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].conv_in.bias, weightMap, "unet.encoders." + idx + ".1.conv_input.bias");
            network.downs.get(i).attns[0].ln1.gamma = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln1.gamma, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_1.weight");
            network.downs.get(i).attns[0].ln1.beta = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln1.beta, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_1.bias");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].attn.qkvLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_1.in_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].attn.oLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_1.out_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].attn.oLinerLayer.bias, weightMap, "unet.encoders." + idx + ".1.attention_1.out_proj.bias");
            network.downs.get(i).attns[0].ln2.gamma = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln2.gamma, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_2.weight");
            network.downs.get(i).attns[0].ln2.beta = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln2.beta, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_2.bias");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].cross_attn.qLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_2.q_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].cross_attn.kLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_2.k_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].cross_attn.vLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_2.v_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].cross_attn.oLinerLayer.weight, weightMap, "unet.encoders." + idx + ".1.attention_2.out_proj.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].cross_attn.oLinerLayer.bias, weightMap, "unet.encoders." + idx + ".1.attention_2.out_proj.bias");
            network.downs.get(i).attns[0].ln3.gamma = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln3.gamma, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_3.weight");
            network.downs.get(i).attns[0].ln3.beta = ClipModelUtils.loadData(network.downs.get(i).attns[0].ln3.beta, weightMap, 1, "unet.encoders." + idx + ".1.layernorm_3.bias");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].geglu1.weight, weightMap, "unet.encoders." + idx + ".1.linear_geglu_1.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].geglu1.bias, weightMap, "unet.encoders." + idx + ".1.linear_geglu_1.bias");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].geglu2.weight, weightMap, "unet.encoders." + idx + ".1.linear_geglu_2.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].geglu2.bias, weightMap, "unet.encoders." + idx + ".1.linear_geglu_2.bias");
            network.downs.get(i).attns[0].conv_out.weight = ClipModelUtils.loadData(network.downs.get(i).attns[0].conv_out.weight, weightMap, 4, "unet.encoders." + idx + ".1.conv_output.weight");
            ClipModelUtils.loadData(network.downs.get(i).attns[0].conv_out.bias, weightMap, "unet.encoders." + idx + ".1.conv_output.bias");
            ClipModelUtils.loadData(network.downs.get(i).down.weight, weightMap, "unet.encoders." + (idx + 1) + ".0.weight");
            ClipModelUtils.loadData(network.downs.get(i).down.bias, weightMap, "unet.encoders." + (idx + 1) + ".0.bias");
        }
        network.down_res.gn_feature.gamma = ClipModelUtils.loadData(network.down_res.gn_feature.gamma, weightMap, 1, "unet.encoders.5.0.groupnorm_feature.weight");
        network.down_res.gn_feature.beta = ClipModelUtils.loadData(network.down_res.gn_feature.beta, weightMap, 1, "unet.encoders.5.0.groupnorm_feature.bias");
        ClipModelUtils.loadData(network.down_res.conv_feature.weight, weightMap, "unet.encoders.5.0.conv_feature.weight");
        ClipModelUtils.loadData(network.down_res.conv_feature.bias, weightMap, "unet.encoders.5.0.conv_feature.bias");
        ClipModelUtils.loadData(network.down_res.temb.linear.weight, weightMap, "unet.encoders.5.0.linear_time.weight");
        ClipModelUtils.loadData(network.down_res.temb.linear.bias, weightMap, "unet.encoders.5.0.linear_time.bias");
        network.down_res.gn_merged.gamma = ClipModelUtils.loadData(network.down_res.gn_merged.gamma, weightMap, 1, "unet.encoders.5.0.groupnorm_merged.weight");
        network.down_res.gn_merged.beta = ClipModelUtils.loadData(network.down_res.gn_merged.beta, weightMap, 1, "unet.encoders.5.0.groupnorm_merged.bias");
        ClipModelUtils.loadData(network.down_res.conv_merged.weight, weightMap, "unet.encoders.5.0.conv_merged.weight");
        ClipModelUtils.loadData(network.down_res.conv_merged.bias, weightMap, "unet.encoders.5.0.conv_merged.bias");
        /**
         * mids

         */
        network.mids.res_head.gn_feature.gamma = ClipModelUtils.loadData(network.mids.res_head.gn_feature.gamma, weightMap, 1, "unet.bottleneck.0.groupnorm_feature.weight");
        network.mids.res_head.gn_feature.beta = ClipModelUtils.loadData(network.mids.res_head.gn_feature.beta, weightMap, 1, "unet.bottleneck.0.groupnorm_feature.bias");
        ClipModelUtils.loadData(network.mids.res_head.conv_feature.weight, weightMap, "unet.bottleneck.0.conv_feature.weight");
        ClipModelUtils.loadData(network.mids.res_head.conv_feature.bias, weightMap, "unet.bottleneck.0.conv_feature.bias");
        ClipModelUtils.loadData(network.mids.res_head.temb.linear.weight, weightMap, "unet.bottleneck.0.linear_time.weight");
        ClipModelUtils.loadData(network.mids.res_head.temb.linear.bias, weightMap, "unet.bottleneck.0.linear_time.bias");
        network.mids.res_head.gn_merged.gamma = ClipModelUtils.loadData(network.mids.res_head.gn_merged.gamma, weightMap, 1, "unet.bottleneck.0.groupnorm_merged.weight");
        network.mids.res_head.gn_merged.beta = ClipModelUtils.loadData(network.mids.res_head.gn_merged.beta, weightMap, 1, "unet.bottleneck.0.groupnorm_merged.bias");
        ClipModelUtils.loadData(network.mids.res_head.conv_merged.weight, weightMap, "unet.bottleneck.0.conv_merged.weight");
        ClipModelUtils.loadData(network.mids.res_head.conv_merged.bias, weightMap, "unet.bottleneck.0.conv_merged.bias");
        /**
         * attn

         */
        network.mids.attns.gn.gamma = ClipModelUtils.loadData(network.mids.attns.gn.gamma, weightMap, 1, "unet.bottleneck.1.groupnorm.weight");
        network.mids.attns.gn.beta = ClipModelUtils.loadData(network.mids.attns.gn.beta, weightMap, 1, "unet.bottleneck.1.groupnorm.bias");
        network.mids.attns.conv_in.weight = ClipModelUtils.loadData(network.mids.attns.conv_in.weight, weightMap, 4, "unet.bottleneck.1.conv_input.weight");
        ClipModelUtils.loadData(network.mids.attns.conv_in.bias, weightMap, "unet.bottleneck.1.conv_input.bias");
        network.mids.attns.ln1.gamma = ClipModelUtils.loadData(network.mids.attns.ln1.gamma, weightMap, 1, "unet.bottleneck.1.layernorm_1.weight");
        network.mids.attns.ln1.beta = ClipModelUtils.loadData(network.mids.attns.ln1.beta, weightMap, 1, "unet.bottleneck.1.layernorm_1.bias");
        ClipModelUtils.loadData(network.mids.attns.attn.qkvLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_1.in_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.attn.oLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_1.out_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.attn.oLinerLayer.bias, weightMap, "unet.bottleneck.1.attention_1.out_proj.bias");
        network.mids.attns.ln2.gamma = ClipModelUtils.loadData(network.mids.attns.ln2.gamma, weightMap, 1, "unet.bottleneck.1.layernorm_2.weight");
        network.mids.attns.ln2.beta = ClipModelUtils.loadData(network.mids.attns.ln2.beta, weightMap, 1, "unet.bottleneck.1.layernorm_2.bias");
        ClipModelUtils.loadData(network.mids.attns.cross_attn.qLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_2.q_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.cross_attn.kLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_2.k_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.cross_attn.vLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_2.v_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.cross_attn.oLinerLayer.weight, weightMap, "unet.bottleneck.1.attention_2.out_proj.weight");
        ClipModelUtils.loadData(network.mids.attns.cross_attn.oLinerLayer.bias, weightMap, "unet.bottleneck.1.attention_2.out_proj.bias");
        network.mids.attns.ln3.gamma = ClipModelUtils.loadData(network.mids.attns.ln3.gamma, weightMap, 1, "unet.bottleneck.1.layernorm_3.weight");
        network.mids.attns.ln3.beta = ClipModelUtils.loadData(network.mids.attns.ln3.beta, weightMap, 1, "unet.bottleneck.1.layernorm_3.bias");
        ClipModelUtils.loadData(network.mids.attns.geglu1.weight, weightMap, "unet.bottleneck.1.linear_geglu_1.weight");
        ClipModelUtils.loadData(network.mids.attns.geglu1.bias, weightMap, "unet.bottleneck.1.linear_geglu_1.bias");
        ClipModelUtils.loadData(network.mids.attns.geglu2.weight, weightMap, "unet.bottleneck.1.linear_geglu_2.weight");
        ClipModelUtils.loadData(network.mids.attns.geglu2.bias, weightMap, "unet.bottleneck.1.linear_geglu_2.bias");
        network.mids.attns.conv_out.weight = ClipModelUtils.loadData(network.mids.attns.conv_out.weight, weightMap, 4, "unet.bottleneck.1.conv_output.weight");
        ClipModelUtils.loadData(network.mids.attns.conv_out.bias, weightMap, "unet.bottleneck.1.conv_output.bias");
        network.mids.res_fail.gn_feature.gamma = ClipModelUtils.loadData(network.mids.res_fail.gn_feature.gamma, weightMap, 1, "unet.bottleneck.2.groupnorm_feature.weight");
        network.mids.res_fail.gn_feature.beta = ClipModelUtils.loadData(network.mids.res_fail.gn_feature.beta, weightMap, 1, "unet.bottleneck.2.groupnorm_feature.bias");
        ClipModelUtils.loadData(network.mids.res_fail.conv_feature.weight, weightMap, "unet.bottleneck.2.conv_feature.weight");
        ClipModelUtils.loadData(network.mids.res_fail.conv_feature.bias, weightMap, "unet.bottleneck.2.conv_feature.bias");
        ClipModelUtils.loadData(network.mids.res_fail.temb.linear.weight, weightMap, "unet.bottleneck.2.linear_time.weight");
        ClipModelUtils.loadData(network.mids.res_fail.temb.linear.bias, weightMap, "unet.bottleneck.2.linear_time.bias");
        network.mids.res_fail.gn_merged.gamma = ClipModelUtils.loadData(network.mids.res_fail.gn_merged.gamma, weightMap, 1, "unet.bottleneck.2.groupnorm_merged.weight");
        network.mids.res_fail.gn_merged.beta = ClipModelUtils.loadData(network.mids.res_fail.gn_merged.beta, weightMap, 1, "unet.bottleneck.2.groupnorm_merged.bias");
        ClipModelUtils.loadData(network.mids.res_fail.conv_merged.weight, weightMap, "unet.bottleneck.2.conv_merged.weight");
        ClipModelUtils.loadData(network.mids.res_fail.conv_merged.bias, weightMap, "unet.bottleneck.2.conv_merged.bias");
        /**
         * ups

         */
        network.up_res.gn_feature.gamma = ClipModelUtils.loadData(network.up_res.gn_feature.gamma, weightMap, 1, "unet.decoders.0.0.groupnorm_feature.weight");
        network.up_res.gn_feature.beta = ClipModelUtils.loadData(network.up_res.gn_feature.beta, weightMap, 1, "unet.decoders.0.0.groupnorm_feature.bias");
        ClipModelUtils.loadData(network.up_res.conv_feature.weight, weightMap, "unet.decoders.0.0.conv_feature.weight");
        ClipModelUtils.loadData(network.up_res.conv_feature.bias, weightMap, "unet.decoders.0.0.conv_feature.bias");
        ClipModelUtils.loadData(network.up_res.temb.linear.weight, weightMap, "unet.decoders.0.0.linear_time.weight");
        ClipModelUtils.loadData(network.up_res.temb.linear.bias, weightMap, "unet.decoders.0.0.linear_time.bias");
        network.up_res.gn_merged.gamma = ClipModelUtils.loadData(network.up_res.gn_merged.gamma, weightMap, 1, "unet.decoders.0.0.groupnorm_merged.weight");
        network.up_res.gn_merged.beta = ClipModelUtils.loadData(network.up_res.gn_merged.beta, weightMap, 1, "unet.decoders.0.0.groupnorm_merged.bias");
        ClipModelUtils.loadData(network.up_res.conv_merged.weight, weightMap, "unet.decoders.0.0.conv_merged.weight");
        ClipModelUtils.loadData(network.up_res.conv_merged.bias, weightMap, "unet.decoders.0.0.conv_merged.bias");
        network.up_res.residual_layer.weight = ClipModelUtils.loadData(network.up_res.residual_layer.weight, weightMap, 4, "unet.decoders.0.0.residual_layer.weight");
        ClipModelUtils.loadData(network.up_res.residual_layer.bias, weightMap, "unet.decoders.0.0.residual_layer.bias");
        /**
         * conv_in

         */
        for (int i = 0; i < 2; i++) {
            int idx = i * 2 + 1;
            ClipModelUtils.loadData(network.ups.get(i).up.conv.weight, weightMap, "unet.decoders." + idx + ".0.conv.weight");
            ClipModelUtils.loadData(network.ups.get(i).up.conv.bias, weightMap, "unet.decoders." + idx + ".0.conv.bias");
            int idx2 = i * 2 + 2;
            /**
             * resnet block

             */
            network.ups.get(i).res[0].gn_feature.gamma = ClipModelUtils.loadData(network.ups.get(i).res[0].gn_feature.gamma, weightMap, 1, "unet.decoders." + idx2 + ".0.groupnorm_feature.weight");
            network.ups.get(i).res[0].gn_feature.beta = ClipModelUtils.loadData(network.ups.get(i).res[0].gn_feature.beta, weightMap, 1, "unet.decoders." + idx2 + ".0.groupnorm_feature.bias");
            ClipModelUtils.loadData(network.ups.get(i).res[0].conv_feature.weight, weightMap, "unet.decoders." + idx2 + ".0.conv_feature.weight");
            ClipModelUtils.loadData(network.ups.get(i).res[0].conv_feature.bias, weightMap, "unet.decoders." + idx2 + ".0.conv_feature.bias");
            ClipModelUtils.loadData(network.ups.get(i).res[0].temb.linear.weight, weightMap, "unet.decoders." + idx2 + ".0.linear_time.weight");
            ClipModelUtils.loadData(network.ups.get(i).res[0].temb.linear.bias, weightMap, "unet.decoders." + idx2 + ".0.linear_time.bias");
            network.ups.get(i).res[0].gn_merged.gamma = ClipModelUtils.loadData(network.ups.get(i).res[0].gn_merged.gamma, weightMap, 1, "unet.decoders." + idx2 + ".0.groupnorm_merged.weight");
            network.ups.get(i).res[0].gn_merged.beta = ClipModelUtils.loadData(network.ups.get(i).res[0].gn_merged.beta, weightMap, 1, "unet.decoders." + idx2 + ".0.groupnorm_merged.bias");
            ClipModelUtils.loadData(network.ups.get(i).res[0].conv_merged.weight, weightMap, "unet.decoders." + idx2 + ".0.conv_merged.weight");
            ClipModelUtils.loadData(network.ups.get(i).res[0].conv_merged.bias, weightMap, "unet.decoders." + idx2 + ".0.conv_merged.bias");
            network.ups.get(i).res[0].residual_layer.weight = ClipModelUtils.loadData(network.ups.get(i).res[0].residual_layer.weight, weightMap, 4, "unet.decoders." + idx2 + ".0.residual_layer.weight");
            ClipModelUtils.loadData(network.ups.get(i).res[0].residual_layer.bias, weightMap, "unet.decoders." + idx2 + ".0.residual_layer.bias");
            /**
             * attn block

             */
            network.ups.get(i).attns[0].gn.gamma = ClipModelUtils.loadData(network.ups.get(i).attns[0].gn.gamma, weightMap, 1, "unet.decoders." + idx2 + ".1.groupnorm.weight");
            network.ups.get(i).attns[0].gn.beta = ClipModelUtils.loadData(network.ups.get(i).attns[0].gn.beta, weightMap, 1, "unet.decoders." + idx2 + ".1.groupnorm.bias");
            network.ups.get(i).attns[0].conv_in.weight = ClipModelUtils.loadData(network.ups.get(i).attns[0].conv_in.weight, weightMap, 4, "unet.decoders." + idx2 + ".1.conv_input.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].conv_in.bias, weightMap, "unet.decoders." + idx2 + ".1.conv_input.bias");
            network.ups.get(i).attns[0].ln1.gamma = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln1.gamma, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_1.weight");
            network.ups.get(i).attns[0].ln1.beta = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln1.beta, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_1.bias");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].attn.qkvLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_1.in_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].attn.oLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_1.out_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].attn.oLinerLayer.bias, weightMap, "unet.decoders." + idx2 + ".1.attention_1.out_proj.bias");
            network.ups.get(i).attns[0].ln2.gamma = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln2.gamma, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_2.weight");
            network.ups.get(i).attns[0].ln2.beta = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln2.beta, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_2.bias");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].cross_attn.qLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_2.q_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].cross_attn.kLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_2.k_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].cross_attn.vLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_2.v_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].cross_attn.oLinerLayer.weight, weightMap, "unet.decoders." + idx2 + ".1.attention_2.out_proj.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].cross_attn.oLinerLayer.bias, weightMap, "unet.decoders." + idx2 + ".1.attention_2.out_proj.bias");
            network.ups.get(i).attns[0].ln3.gamma = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln3.gamma, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_3.weight");
            network.ups.get(i).attns[0].ln3.beta = ClipModelUtils.loadData(network.ups.get(i).attns[0].ln3.beta, weightMap, 1, "unet.decoders." + idx2 + ".1.layernorm_3.bias");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].geglu1.weight, weightMap, "unet.decoders." + idx2 + ".1.linear_geglu_1.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].geglu1.bias, weightMap, "unet.decoders." + idx2 + ".1.linear_geglu_1.bias");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].geglu2.weight, weightMap, "unet.decoders." + idx2 + ".1.linear_geglu_2.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].geglu2.bias, weightMap, "unet.decoders." + idx2 + ".1.linear_geglu_2.bias");
            network.ups.get(i).attns[0].conv_out.weight = ClipModelUtils.loadData(network.ups.get(i).attns[0].conv_out.weight, weightMap, 4, "unet.decoders." + idx2 + ".1.conv_output.weight");
            ClipModelUtils.loadData(network.ups.get(i).attns[0].conv_out.bias, weightMap, "unet.decoders." + idx2 + ".1.conv_output.bias");
        }
        ClipModelUtils.loadData(network.conv_out.weight, weightMap, "unet.decoders.5.0.weight");
        ClipModelUtils.loadData(network.conv_out.bias, weightMap, "unet.decoders.5.0.bias");
        network.norm.gamma = ClipModelUtils.loadData(network.norm.gamma, weightMap, 1, "final.groupnorm.weight");
        network.norm.beta = ClipModelUtils.loadData(network.norm.beta, weightMap, 1, "final.groupnorm.bias");
        ClipModelUtils.loadData(network.conv_final.weight, weightMap, "final.conv.weight");
        ClipModelUtils.loadData(network.conv_final.bias, weightMap, "final.conv.bias");
    }

    public static void main(String[] args) {
        int batchSize = 2;
        int channel = 4;
        int height = 32;
        int width = 32;
        //		int oChannel = 128;
        int headNum = 8;
        int nTime = 512;
        int timeSteps = 1000;
        int contextTime = 64;
        int contextDim = 512;
        int[] downChannels = new int[]{64, 128, 256};
        Transformer tf = new Transformer();
        tf.updater = UpdaterType.adamw;
        tf.CUDNN = true;
        tf.learnRate = 0.001f;
        tf.RUN_MODEL = RunModel.TRAIN;
        tf.number = batchSize;
        float[] data = RandomUtils.order(batchSize * channel * height * width, 0.01f, 0.01f);
        Tensor input = new Tensor(batchSize, channel, height, width, data, true);
        Tensor t = new Tensor(batchSize, 1, 1, 1, new float[]{10, 214}, true);
        float[] delta_data = RandomUtils.order(batchSize * channel * height * width, 0.01f, 0.01f);
        Tensor delta = new Tensor(batchSize, channel, height, width, delta_data, true);
        Tensor context = new Tensor(batchSize * contextTime, 1, 1, contextDim, MatrixUtils.order(batchSize * contextTime * contextDim, 0.01f, 0.01f), true);
        //		context.showDM();
        UNetCond2 unet = new UNetCond2(channel, height, width, downChannels, headNum, nTime, timeSteps, contextTime, contextDim, 1, 32, tf);
        String weight = "H:\\model\\unet2.json";
        loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), unet, true);
        for (int i = 0; i < 2; i++) {
            tf.train_time++;
            JCuda.cudaDeviceSynchronize();
            unet.forward(input, t, context);
            unet.getOutput().showDM("output");
            unet.back(delta);
            unet.diff.showDM("diff");
            if (i == 0) {
                unet.update();
            }
            tf.clearCacheDelta();
        }
    }

    public void initLayers() {
        t_embd = new TimeEmbeddingLayer(timeSteps, nTime, nTime, true, network);
        conv_in = new ConvolutionLayer(channel, downChannels[0], width, height, 3, 3, 1, 1, true, network);
        conv_in.setName("conv_in");
        downs = new ArrayList<UNetDownBlock>(downChannels.length - 1);
        ups = new ArrayList<UNetUpBlock>(downChannels.length - 1);
        cats = new ArrayList<RouteLayer>(downChannels.length - 1);
        int ih = height;
        int iw = width;
        Stack<Layer> downLayers = new Stack<Layer>();
        for (int i = 0; i < downChannels.length - 1; i++) {
            UNetDownBlock down = new UNetDownBlock("down-" + i, downChannels[i], downChannels[i + 1], ih, iw, nTime, headNum, contextTime, contextDim, groupNum, numLayer, network);
            downs.add(down);
            down.pushStack(downLayers);
            ih = down.oHeight;
            iw = down.oWidth;
        }
        int oc = downChannels[downChannels.length - 1];
        down_res = new UNetResidualBlock(oc, oc, ih, iw, nTime, groupNum, network);
        down_res.setName("down_res");
        mids = new UNetMidBlock(oc, oc, ih, iw, nTime, headNum, contextTime, contextDim, groupNum, network);
        mids.setName("mid");
        cat_res = new RouteLayer(new Layer[]{mids, down_res});
        up_res = new UNetResidualBlock(oc * 2, oc, ih, iw, nTime, groupNum, network);
        up_res.setName("up_res");
        cats.add(new RouteLayer(new Layer[]{up_res, downs.get(downs.size() - 1)}));
        for (int i = downChannels.length - 2; i >= 0; i--) {
            UNetUpBlock up = new UNetUpBlock("ups-" + i, downChannels[i + 1], downChannels[i + 1], downChannels[i], ih, iw, nTime, headNum, contextTime, contextDim, groupNum, numLayer, downLayers, network);
            ups.add(up);
            ih = up.oHeight;
            iw = up.oWidth;
            if (i > 0) {
                cats.add(new RouteLayer(new Layer[]{up, downs.get(i - 1)}));
            }
        }
        cat_out = new RouteLayer(new Layer[]{ups.get(ups.size() - 1), conv_in});
        conv_out = new ConvolutionLayer(downChannels[0] * 2, downChannels[0], width, height, 3, 3, 1, 1, true, network);
        conv_out.setName("conv_out");
        norm = new GNLayer(groupNum, conv_out, BNType.conv_bn);
        act = new SiLULayer(norm);
        conv_final = new ConvolutionLayer(downChannels[0], channel, width, height, 3, 3, 1, 1, true, network);
    }

    @Override
    public void init() {
        // TODO Auto-generated method stub
        this.number = this.network.number;
    }

    public void init(Tensor input) {
        // TODO Auto-generated method stub
        this.number = input.number;
    }

    @Override
    public void initBack() {
        // TODO Auto-generated method stub
        if (tDiff == null || tDiff.number != this.number) {
            tDiff = Tensor.createGPUTensor(tDiff, this.number, 1, 1, nTime, true);
        } else {
            tDiff.clearGPU();
        }
    }

    @Override
    public void initParam() {
        // TODO Auto-generated method stub
    }

    @Override
    public void output() {
        // TODO Auto-generated method stub
    }

    public void output(Tensor time) {
        // TODO Auto-generated method stub
        t_embd.forward(time);
        conv_in.forward(input);
        Tensor x = conv_in.getOutput();
        Tensor t_x = t_embd.getOutput();
        //		t_x.showDM("in---t");
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).forward(x, t_x);
            x = downs.get(i).getOutput();
            //			x.showDM("donwx:"+i);
        }
        //		x.showDM("donwx");
        down_res.forward(x, t_x);
        mids.forward(down_res.getOutput(), t_x);
        //		mids.getOutput().showDM("mids");
        cat_res.forward();
        up_res.forward(cat_res.getOutput(), t_x);
        for (int i = 0; i < ups.size(); i++) {
            cats.get(i).forward();
            x = cats.get(i).getOutput();
            ups.get(i).forward(x, t_x);
        }
        cat_out.forward();
        //		cat_out.getOutput().showDM("up-out");
        conv_out.forward(cat_out.getOutput());
        //		conv_out.getOutput().showDM("conv_out");
        norm.forward(conv_out.getOutput());
        //		norm.getOutput().showDM("up");
        act.forward(norm.getOutput());
        conv_final.forward(act.getOutput());
        this.output = conv_final.getOutput();
    }

    public void output(Tensor time, Tensor context) {
        // TODO Auto-generated method stub
        t_embd.forward(time);
        conv_in.forward(input);
        Tensor x = conv_in.getOutput();
        Tensor t_x = t_embd.getOutput();
        //		t_x.showDM("in---t");
        //		x.showDM("x");
        //		System.err.println("start");
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).forward(x, t_x, context);
            x = downs.get(i).getOutput();
            //			x.showDM("donwx:"+i);
        }
        //		System.err.println("down");
        //		x.showDM("donwx");
        down_res.forward(x, t_x);
        mids.forward(down_res.getOutput(), t_x, context);
        //		System.err.println("mids");
        //		mids.getOutput().showDM("mids");
        cat_res.forward();
        up_res.forward(cat_res.getOutput(), t_x);
        for (int i = 0; i < ups.size(); i++) {
            cats.get(i).forward();
            x = cats.get(i).getOutput();
            ups.get(i).forward(x, t_x, context);
        }
        //		System.err.println("up");
        cat_out.forward();
        //		cat_out.getOutput().showDM("up-out");
        conv_out.forward(cat_out.getOutput());
        //		conv_out.getOutput().showDM("conv_out");
        norm.forward(conv_out.getOutput());
        //		norm.getOutput().showDM("up");
        act.forward(norm.getOutput());
        conv_final.forward(act.getOutput());
        this.output = conv_final.getOutput();
        //		this.output.showDM("out");
    }

    @Override
    public Tensor getOutput() {
        // TODO Auto-generated method stub
        return this.output;
    }

    @Override
    public void diff() {
        // TODO Auto-generated method stub
        conv_final.back(delta);
        act.back(conv_final.diff);
        norm.back(act.diff);
        conv_out.back(norm.diff);
        cat_out.back(conv_out.diff);
        for (int i = ups.size() - 1; i >= 0; i--) {
            ups.get(i).back(ups.get(i).cache_delta, tDiff);
            cats.get(i).back(ups.get(i).diff);
        }
        up_res.back(up_res.cache_delta, tDiff);
        cat_res.back(up_res.diff);
        mids.back(mids.cache_delta, tDiff);
        down_res.back(mids.diff, tDiff);
        Tensor d = down_res.diff;
        //		d.showDM("mids.diff");
        for (int i = downs.size() - 1; i >= 0; i--) {
            downs.get(i).back(d, tDiff);
            d = downs.get(i).diff;
        }
        conv_in.back(d);
        //		tDiff.showDM("tDiff");
        t_embd.back(tDiff);
        this.diff = conv_in.diff;
    }

    @Override
    public void forward() {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init();
        /**
         * 设置输入

         */
        this.setInput();
        /**
         * 计算输出

         */
        this.output();
    }

    @Override
    public void back() {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta();
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void backTemp() {
        // TODO Auto-generated method stub
    }

    @Override
    public void forward(Tensor input) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output();
    }

    public void forward(Tensor input, Tensor time) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(time);
    }

    public void forward(Tensor input, Tensor time, Tensor context) {
        // TODO Auto-generated method stub
        /**
         * 参数初始化

         */
        this.init(input);
        /**
         * 设置输入

         */
        this.setInput(input);
        /**
         * 计算输出

         */
        this.output(time, context);
    }

    @Override
    public void back(Tensor delta) {
        // TODO Auto-generated method stub
        initBack();
        /**
         * 设置梯度

         */
        this.setDelta(delta);
        /**
         * 计算梯度

         */
        this.diff();
    }

    @Override
    public void update() {
        // TODO Auto-generated method stub
        t_embd.update();
        conv_in.update();
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).update();
        }
        down_res.update();
        mids.update();
        up_res.update();
        for (int i = 0; i < ups.size(); i++) {
            ups.get(i).update();
        }
        conv_out.update();
        norm.update();
        conv_final.update();
    }

    @Override
    public void showDiff() {
        // TODO Auto-generated method stub
    }

    @Override
    public LayerType getLayerType() {
        // TODO Auto-generated method stub
        return LayerType.unet_down;
    }

    @Override
    public float[][][][] output(float[][][][] input) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void initCache() {
        // TODO Auto-generated method stub
    }

    @Override
    public void accGrad(float scale) {
        // TODO Auto-generated method stub
        t_embd.accGrad(scale);
        conv_in.accGrad(scale);
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).accGrad(scale);
        }
        down_res.accGrad(scale);
        mids.accGrad(scale);
        up_res.accGrad(scale);
        for (int i = 0; i < ups.size(); i++) {
            ups.get(i).accGrad(scale);
        }
        conv_out.accGrad(scale);
        norm.accGrad(scale);
        conv_final.accGrad(scale);
    }

    public void saveModel(RandomAccessFile outputStream) throws IOException {
        t_embd.saveModel(outputStream);
        conv_in.saveModel(outputStream);
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).saveModel(outputStream);
        }
        down_res.saveModel(outputStream);
        mids.saveModel(outputStream);
        up_res.saveModel(outputStream);
        for (int i = 0; i < ups.size(); i++) {
            ups.get(i).saveModel(outputStream);
        }
        conv_out.saveModel(outputStream);
        norm.saveModel(outputStream);
        conv_final.saveModel(outputStream);
    }

    public void loadModel(RandomAccessFile inputStream) throws IOException {
        t_embd.loadModel(inputStream);
        conv_in.loadModel(inputStream);
        for (int i = 0; i < downs.size(); i++) {
            downs.get(i).loadModel(inputStream);
        }
        down_res.loadModel(inputStream);
        mids.loadModel(inputStream);
        up_res.loadModel(inputStream);
        for (int i = 0; i < ups.size(); i++) {
            ups.get(i).loadModel(inputStream);
        }
        conv_out.loadModel(inputStream);
        norm.loadModel(inputStream);
        conv_final.loadModel(inputStream);
    }
}

