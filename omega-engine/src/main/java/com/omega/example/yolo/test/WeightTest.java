package com.omega.example.yolo.test;

import jcuda.Sizeof;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WeightTest {
    public static void main(String[] args) {
        try (RandomAccessFile file = new RandomAccessFile("H:\\voc\\yolo-weights\\yolov3-tiny.weights", "r")) {
            int major = readInt(file);
            int minor = readInt(file);
            int revision = readInt(file);
            long seen = readBigInt(file);
            System.out.println("major:" + major);
            System.out.println("minor:" + minor);
            System.out.println("revision:" + revision);
            System.out.println("seen:" + seen);
            for (int i = 0; i < 10000; i++) {
                System.out.println(readFloat(file));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static long readBigInt(RandomAccessFile inputStream) throws IOException {
        long retVal;
        byte[] buffer = new byte[Sizeof.LONG];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getLong();
        return retVal;
    }

    public static int readInt(RandomAccessFile inputStream) throws IOException {
        int retVal;
        byte[] buffer = new byte[Sizeof.INT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getInt();
        return retVal;
    }

    public static float readFloat(RandomAccessFile inputStream) throws IOException {
        float retVal;
        byte[] buffer = new byte[Sizeof.FLOAT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getFloat();
        return retVal;
    }

    public static void readFloat(RandomAccessFile inputStream, float[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            data[i] = readFloat(inputStream);
        }
    }
}

