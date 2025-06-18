package com.omega.example.transformer.utils;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.network.*;
import com.omega.engine.nn.network.vae.TinyVQVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE2;
import com.omega.engine.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

public class ModelUtils {
	
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
