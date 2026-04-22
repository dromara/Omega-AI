package com.omega.example.transformer.utils;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.network.ASR;
import com.omega.engine.nn.network.ClipTextModel;
import com.omega.engine.nn.network.DiffusionUNetCond2;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.Llava;
import com.omega.engine.nn.network.NanoGPT;
import com.omega.engine.nn.network.OpenSoraDIT;
import com.omega.engine.nn.network.T5Encoder;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.nn.network.dit.DiT_ORG;
import com.omega.engine.nn.network.dit.DiT_ORG2;
import com.omega.engine.nn.network.dit.DiT_ORG_SRA;
import com.omega.engine.nn.network.dit.DiT_TXT;
import com.omega.engine.nn.network.dit.Dinov2;
import com.omega.engine.nn.network.dit.FluxDiT;
import com.omega.engine.nn.network.dit.FluxDiT2;
import com.omega.engine.nn.network.dit.FluxDiT3;
import com.omega.engine.nn.network.dit.FluxDiT_REG;
import com.omega.engine.nn.network.dit.FluxDiT_REPA;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT3;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT4;
import com.omega.engine.nn.network.dit.FluxDiT_SPRINT_REG;
import com.omega.engine.nn.network.dit.FluxDiT_TREAD;
import com.omega.engine.nn.network.dit.JiT;
import com.omega.engine.nn.network.dit.JiT_REPA;
import com.omega.engine.nn.network.dit.MMDiT;
import com.omega.engine.nn.network.dit.MMDiT_RoPE;
import com.omega.engine.nn.network.dit.OmegaDiT;
import com.omega.engine.nn.network.dit.OmegaDiTFullLabel;
import com.omega.engine.nn.network.dit.PixArtDiT;
import com.omega.engine.nn.network.dit.SanaDiT;
import com.omega.engine.nn.network.vae.LTXVideo_VAE;
import com.omega.engine.nn.network.vae.TinyVQVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.nn.network.vae.WFVAE;
import com.omega.engine.nn.network.video.OmegaVideo;
import com.omega.engine.tensor.Tensor;

public class ModelUtils {
	
	public static void saveModel(OmegaVideo model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(OmegaVideo model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(LTXVideo_VAE model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(LTXVideo_VAE model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(OmegaDiTFullLabel model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(OmegaDiTFullLabel model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_SPRINT_REG model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_SPRINT_REG model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(T5Encoder model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(T5Encoder model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_TREAD model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_TREAD model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	
	public static void saveModel(FluxDiT_SPRINT4 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_SPRINT4 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_SPRINT3 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_SPRINT3 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(OmegaDiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(OmegaDiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_SPRINT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_SPRINT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(JiT_REPA model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(JiT_REPA model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(JiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(JiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_REG model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_REG model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT_REPA model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT_REPA model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(Dinov2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(Dinov2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT3 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT3 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(FluxDiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(FluxDiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(SanaDiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(SanaDiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(ClipTextModel model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(ClipTextModel model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(DiT_TXT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(DiT_TXT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(Yolo model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(Yolo model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(PixArtDiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(PixArtDiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(OpenSoraDIT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(OpenSoraDIT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(MMDiT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(MMDiT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(MMDiT_RoPE model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(MMDiT_RoPE model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(DiT_ORG_SRA model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(DiT_ORG_SRA model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(DiT_ORG2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(DiT_ORG2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void saveModel(WFVAE model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
	public static void loadModel(WFVAE model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
	
    public static void saveModel(Llama2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(TinyVQVAE2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(TinyVQVAE2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
    
    public static void saveModel(DiT_ORG model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    
    public static void loadModel(DiT_ORG model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(TinyVQVAE model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(TinyVQVAE model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(Llama2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(Llama3 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(Llama3 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(Llava model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(Llava model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadPertrainModel(Llava model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadPertrainModel(File);
            initParams(model.getDecoder().getVersionProj().weight, 0.0f, 0.02f);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void initParams(Tensor p, float mean, float std) {
        p.setData(RandomUtils.normal_(p.dataLength, mean, std));
    }

    public static void saveModel(NanoGPT model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile aFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(aFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(NanoGPT model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(DiffusionUNetCond2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(DiffusionUNetCond2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(VQVAE2 model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(VQVAE2 model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void saveModel(ASR model, String outpath) {
        File file = new File(outpath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try (RandomAccessFile rFile = new RandomAccessFile(file, "rw")) {
            System.out.println("start save model...");
            model.saveModel(rFile);
            System.out.println("model save success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void loadModel(ASR model, String inputPath) {
        try (RandomAccessFile File = new RandomAccessFile(inputPath, "r")) {
            System.out.println("start load model...");
            model.loadModel(File);
            System.out.println("model load success...");
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }
}
