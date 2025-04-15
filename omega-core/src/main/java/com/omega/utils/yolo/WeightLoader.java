package com.omega.utils.yolo;

import java.io.*;

public class WeightLoader {
    public static byte[] readFromByteFile(String pathname) throws IOException {
        File filename = new File(pathname);
        BufferedInputStream in = new BufferedInputStream(new FileInputStream(filename));
        ByteArrayOutputStream out = new ByteArrayOutputStream(1024);
        byte[] temp = new byte[1024];
        int size = 0;
        while ((size = in.read(temp)) != -1) {
            out.write(temp, 0, size);
        }
        in.close();
        byte[] content = out.toByteArray();
        return content;
    }

    public static void main(String[] args) {
        String pathname = "H:/voc/train/resnet18.weights";
        try {
            byte[] texts = readFromByteFile(pathname);
            String text = new String(texts, "utf-8");
            System.out.println(text);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}

