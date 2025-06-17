package com.omega.example.clip.utils;

import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.clip.bert.BertLayer;
import com.omega.engine.nn.network.ClipText;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.tensor.Tensor;

import java.util.List;
import java.util.Map;

public class ClipModelUtils {
    public static void loadWeight(Map<String, Object> weightMap, ClipTextModel network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * embeddings

         */
        loadData(network.clip.getEmbeddings().getTokenEmbedding().weight, weightMap, "text_model.embeddings.token_embedding.weight");
        loadData(network.clip.getEmbeddings().getPositionEmbedding().weight, weightMap, "text_model.embeddings.position_embedding.weight");
        /**
         * encoders

         */
        for (int i = 0; i < 12; i++) {
            /**
             * self attention

             */
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.q_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.q_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.k_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.k_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.v_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.v_proj.bias");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, "text_model.encoder.layers." + i + ".self_attn.out_proj.weight");
            loadData(network.clip.getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, "text_model.encoder.layers." + i + ".self_attn.out_proj.bias");
            /**
             * layer norm1

             */
            network.clip.getEncoders().get(i).getNorm1().gamma = loadData(network.clip.getEncoders().get(i).getNorm1().gamma, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm1.weight");
            network.clip.getEncoders().get(i).getNorm1().beta = loadData(network.clip.getEncoders().get(i).getNorm1().beta, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm1.bias");
            /**
             * mlp

             */
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().weight, weightMap, "text_model.encoder.layers." + i + ".mlp.fc1.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear1().bias, weightMap, "text_model.encoder.layers." + i + ".mlp.fc1.bias");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().weight, weightMap, "text_model.encoder.layers." + i + ".mlp.fc2.weight");
            loadData(network.clip.getEncoders().get(i).getMlp().getLinear2().bias, weightMap, "text_model.encoder.layers." + i + ".mlp.fc2.bias");
            /**
             * layer norm2

             */
            network.clip.getEncoders().get(i).getNorm2().gamma = loadData(network.clip.getEncoders().get(i).getNorm2().gamma, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm2.weight");
            network.clip.getEncoders().get(i).getNorm2().beta = loadData(network.clip.getEncoders().get(i).getNorm2().beta, weightMap, 1, "text_model.encoder.layers." + i + ".layer_norm2.bias");
        }
        /**
         * post_layernorm

         */
        network.clip.getFinalLayerNorm().gamma = loadData(network.clip.getFinalLayerNorm().gamma, weightMap, 1, "text_model.final_layer_norm.weight");
        network.clip.getFinalLayerNorm().beta = loadData(network.clip.getFinalLayerNorm().beta, weightMap, 1, "text_model.final_layer_norm.bias");
    }

    public static void loadWeight(Map<String, Object> weightMap, ClipVision network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * embeddings

         */
        loadData(network.getEncoder().getEmbeddings().getClassEmbedding(), weightMap, "embeddings.class_embedding");
        loadData(network.getEncoder().getEmbeddings().getPatchEmbedding().weight, weightMap, "embeddings.patch_embedding.weight");
        loadData(network.getEncoder().getEmbeddings().getPositionEmbedding().weight, weightMap, "embeddings.position_embedding.weight");
        /**
         * pre_layernorm

         */
        network.getEncoder().getPreLayrnorm().gamma = loadData(network.getEncoder().getPreLayrnorm().gamma, weightMap, 1, "pre_layrnorm.weight");
        network.getEncoder().getPreLayrnorm().beta = loadData(network.getEncoder().getPreLayrnorm().beta, weightMap, 1, "pre_layrnorm.bias");
        /**
         * encoders

         */
        for (int i = 0; i < 12; i++) {
            /**
             * attn

             */
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.q_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.q_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.k_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.k_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.v_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.v_proj.bias");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, "encoder.layers." + i + ".self_attn.out_proj.weight");
            loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, "encoder.layers." + i + ".self_attn.out_proj.bias");
            /**
             * ln1

             */
            network.getEncoder().getEncoders().get(i).getNorm1().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm1().gamma, weightMap, 1, "encoder.layers." + i + ".layer_norm1.weight");
            network.getEncoder().getEncoders().get(i).getNorm1().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm1().beta, weightMap, 1, "encoder.layers." + i + ".layer_norm1.bias");
            /**
             * mlp

             */
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().weight, weightMap, "encoder.layers." + i + ".mlp.fc1.weight");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().bias, weightMap, "encoder.layers." + i + ".mlp.fc1.bias");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().weight, weightMap, "encoder.layers." + i + ".mlp.fc2.weight");
            loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().bias, weightMap, "encoder.layers." + i + ".mlp.fc2.bias");
            /**
             * ln2

             */
            network.getEncoder().getEncoders().get(i).getNorm2().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm2().gamma, weightMap, 1, "encoder.layers." + i + ".layer_norm2.weight");
            network.getEncoder().getEncoders().get(i).getNorm2().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm2().beta, weightMap, 1, "encoder.layers." + i + ".layer_norm2.bias");
            //			network.getEncoder().getEncoders().get(i).getNorm2().gamma.showShape();
        }
        /**
         * post_layernorm

         */
        network.getEncoder().getPostLayernorm().gamma = loadData(network.getEncoder().getPostLayernorm().gamma, weightMap, 1, "post_layernorm.weight");
        network.getEncoder().getPostLayernorm().beta = loadData(network.getEncoder().getPostLayernorm().beta, weightMap, 1, "post_layernorm.bias");
    }

    public static void loadWeight(Map<String, Object> weightMap, ClipText network, boolean showLayers) {
        if (showLayers) {
            for (String key : weightMap.keySet()) {
                System.out.println(key);
            }
        }
        /**
         * text_projection

         */
        loadData(network.textProjection, weightMap, "text_projection");
        /**
         * bert.embeddings

         */
        loadData(network.bert.embeddings.wordEmbeddings.weight, weightMap, "bert.embeddings.word_embeddings.weight");
        loadData(network.bert.embeddings.positionEmbeddings.weight, weightMap, "bert.embeddings.position_embeddings.weight");
        loadData(network.bert.embeddings.tokenTypeEmbeddings.weight, weightMap, "bert.embeddings.token_type_embeddings.weight");
        network.bert.embeddings.norm.gamma = loadData(network.bert.embeddings.norm.gamma, weightMap, 1, "bert.embeddings.LayerNorm.weight");
        network.bert.embeddings.norm.beta = loadData(network.bert.embeddings.norm.beta, weightMap, 1, "bert.embeddings.LayerNorm.bias");
        /**
         * bert.encoder

         */
        for (int i = 0; i < 12; i++) {
            BertLayer bl = network.bert.encoder.layers.get(i);
            /**
             * attention

             */
            loadData(bl.attn.attn.getqLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.query.weight");
            loadData(bl.attn.attn.getqLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.query.bias");
            loadData(bl.attn.attn.getkLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.key.weight");
            loadData(bl.attn.attn.getkLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.key.bias");
            loadData(bl.attn.attn.getvLinerLayer().weight, weightMap, "bert.encoder.layer." + i + ".attention.self.value.weight");
            loadData(bl.attn.attn.getvLinerLayer().bias, weightMap, "bert.encoder.layer." + i + ".attention.self.value.bias");
            /**
             * attention output

             */
            loadData(bl.attn.out.linear.weight, weightMap, "bert.encoder.layer." + i + ".attention.output.dense.weight");
            loadData(bl.attn.out.linear.bias, weightMap, "bert.encoder.layer." + i + ".attention.output.dense.bias");
            bl.attn.out.norm.gamma = loadData(bl.attn.out.norm.gamma, weightMap, 1, "bert.encoder.layer." + i + ".attention.output.LayerNorm.weight");
            bl.attn.out.norm.beta = loadData(bl.attn.out.norm.beta, weightMap, 1, "bert.encoder.layer." + i + ".attention.output.LayerNorm.bias");
            /**
             * intermediate

             */
            loadData(bl.inter.linear.weight, weightMap, "bert.encoder.layer." + i + ".intermediate.dense.weight");
            loadData(bl.inter.linear.bias, weightMap, "bert.encoder.layer." + i + ".intermediate.dense.bias");
            /**
             * output

             */
            loadData(bl.out.linear.weight, weightMap, "bert.encoder.layer." + i + ".output.dense.weight");
            loadData(bl.out.linear.bias, weightMap, "bert.encoder.layer." + i + ".output.dense.bias");
            bl.out.norm.gamma = loadData(bl.out.norm.gamma, weightMap, 1, "bert.encoder.layer." + i + ".output.LayerNorm.weight");
            bl.out.norm.beta = loadData(bl.out.norm.beta, weightMap, 1, "bert.encoder.layer." + i + ".output.LayerNorm.bias");
        }
    }

    @SuppressWarnings("unchecked")
    public static void loadData(Tensor x, Map<String, Object> weightMap, String key) {
        Object meta = weightMap.get(key);
        if (meta != null) {
            int dim = getDim(x);
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.getData()[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.getData()[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.setData(MatrixUtils.transform(data));
            }else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.getData()[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }else {
        	System.err.println(key+" is null.");
        }
    }
    
    public static void loadData(Tensor x, Map<String, Object> weightMap, String key,int dim) {
        Object meta = weightMap.get(key);
        if (meta != null) {
        	if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                for (int n = 0; n < dataA.size(); n++) {
                    x.getData()[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                //				x.showShape();
                //				System.out.println(dataA.size()+":"+dataA.get(0).size());
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.getData()[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.setData(MatrixUtils.transform(data));
            } else if (dim == 5) {
            	List<List<List<List<List<Double>>>>> data = (List<List<List<List<List<Double>>>>>) meta;
                x.setData(MatrixUtils.transform(data));
            }  else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                x.getData()[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
        }
    }

    @SuppressWarnings("unchecked")
    public static Tensor loadData(Tensor x, Map<String, Object> weightMap, int dim, String key) {
        Object meta = weightMap.get(key);
        if (meta != null) {
            if (dim == 1) {
                List<Double> dataA = (List<Double>) meta;
                x = new Tensor(1, 1, 1, dataA.size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    x.getData()[n] = dataA.get(n).floatValue();
                }
            } else if (dim == 2) {
                List<List<Double>> dataA = (List<List<Double>>) meta;
                x = new Tensor(dataA.size(), 1, 1, dataA.get(0).size(), true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int w = 0; w < dataA.get(n).size(); w++) {
                        x.getData()[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
                    }
                }
                //				float[][] data = (float[][]) meta;
                //				x.data = MatrixUtils.transform(data);
            } else if (dim == 3) {
                float[][][] data = (float[][][]) meta;
                x.setData(MatrixUtils.transform(data));
            } else {
                List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
                int N = dataA.size();
                int C = dataA.get(0).size();
                int H = dataA.get(0).get(0).size();
                int W = dataA.get(0).get(0).get(0).size();
                x = new Tensor(N, C, H, W, true);
                for (int n = 0; n < dataA.size(); n++) {
                    for (int c = 0; c < dataA.get(n).size(); c++) {
                        for (int h = 0; h < dataA.get(n).get(c).size(); h++) {
                            for (int w = 0; w < dataA.get(n).get(c).get(h).size(); w++) {
                                x.getData()[n * x.getOnceSize() + c * x.getShape()[2] * x.getShape()[3] + h * x.getShape()[3] + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
                            }
                        }
                    }
                }
            }
            x.hostToDevice();
            System.out.println(key + "_finish.");
            return x;
        }
        return null;
    }

    public static int getDim(Tensor x) {
        int dim = 0;
        if (x.getShape()[0] > 1) {
            dim++;
        }
        if (x.getShape()[1] > 1) {
            dim++;
        }
        if (x.getShape()[2] > 1) {
            dim++;
        }
        if (x.getShape()[3] > 1) {
            dim++;
        }
        return dim;
    }
}

