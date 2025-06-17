package com.omega.example.transformer.utils;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.network.utils.ModelUtils;

import java.io.RandomAccessFile;

public class BinFileUtils {
    public static void loadBin(String inputPath) {
        int[] cache = new int[512];
        try (RandomAccessFile file = new RandomAccessFile(inputPath, "r")) {
            ModelUtils.readShort2Int(file, cache);
            System.out.println(JsonUtils.toJson(cache));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        String filePath = "H:\\model\\pretrain_data_6400.bin";
        loadBin(filePath);
    }
}

