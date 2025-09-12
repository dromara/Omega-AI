package com.omega.engine.nn.network.utils;

import jcuda.Sizeof;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.stream.IntStream;

import com.omega.engine.tensor.Tensor;

public class ModelUtils {
    public static void saveIntData(RandomAccessFile outputStream, int[] data) throws IOException {
        writeInt(outputStream, data);
    }

    public static void loadIntData(RandomAccessFile inputStream, int[] data) throws IOException {
        readInt(inputStream, data);
    }

    public static void saveParams(RandomAccessFile outputStream, Tensor data) throws IOException {
        writeFloat(outputStream, data);
    }

    public static void loadParams(RandomAccessFile inputStream, Tensor data) throws IOException {
        readFloat(inputStream, data);
    }

//    public static void readFloat(RandomAccessFile inputStream, Tensor data) throws IOException {
//        for (int i = 0; i < data.data.length; i++) {
//            float v = readFloat(inputStream);
//            data.data[i] = v;
//            if (v == Float.NaN) {
//                System.err.println(v);
//            }
//        }
//        if (data.isHasGPU()) {
//            data.hostToDevice();
//        }
//    }

    /**
     * 使用buffer,批量读取二进制文件数据
     * 对应writeFloat
     * 保存bytes采用小端排序
     *
     * @param inputStream
     * @param data
     * @throws IOException
     */
    public static void readFloat(RandomAccessFile inputStream, Tensor data) throws IOException {
        final int CHUNK_FLOATS = 16384;
        final int CHUNK_BYTES = CHUNK_FLOATS * 4;
        byte[] buffer = new byte[CHUNK_BYTES];

        int totalFloats = data.data.length;
        int processed = 0;

        while (processed < totalFloats) {
            int remainingFloats = totalFloats - processed;
            int floatsToRead = Math.min(remainingFloats, CHUNK_FLOATS);
            int bytesToRead = floatsToRead * 4;

            // 使用readFully确保读取完整数据
            if (bytesToRead > 0) {
                inputStream.readFully(buffer, 0, bytesToRead);

                // 处理数据
                for (int i = 0; i < floatsToRead; i++) {
                    int pos = i * 4;
                    int fbit = (buffer[pos] & 0xFF) |
                            ((buffer[pos + 1] & 0xFF) << 8) |
                            ((buffer[pos + 2] & 0xFF) << 16) |
                            ((buffer[pos + 3] & 0xFF) << 24);

                    data.data[processed + i] = Float.intBitsToFloat(fbit);
                }

                processed += floatsToRead;
            }
        }

        if (data.isHasGPU()) {
            data.hostToDevice();
        }
    }

    public static void readFloat(RandomAccessFile inputStream, float[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            float v = readFloat(inputStream);
            data[i] = v;
            if (v == Float.NaN) {
                System.err.println(v);
            }
        }
    }

    public static void readInt(RandomAccessFile inputStream, int[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            int v = readInt(inputStream);
            data[i] = v;
            if (v == Float.NaN) {
                System.err.println(v);
            }
        }
    }

    public static void readIntLine(String txt, int[] data) throws IOException {
        String[] bs = txt.split("");
        byte[] buffer = new byte[Sizeof.INT];
        int idx = 0;
        for (int i = 0; i < bs.length; i++) {
            buffer[i % 4] = Byte.parseByte(bs[i]);
            if (i + 1 % 4 == 0) {
                int s = readInt(buffer);
                data[idx] = s;
                idx++;
            }
        }
        /**
         * padding
         */
        for (int j = idx; j < data.length; j++) {
            data[j] = 0;
        }
    }

    public static void readShort(RandomAccessFile inputStream, short[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            short v = readShort(inputStream);
            data[i] = v;
            if (v == Float.NaN) {
                System.err.println(v);
            }
        }
    }

    public static void readShort2Int(RandomAccessFile inputStream, int[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            short v = readShort(inputStream);
            data[i] = v;
            if (v == Float.NaN) {
                System.err.println(v);
            }
        }
    }

    public static void readShort2Int(RandomAccessFile inputStream, int[] data, short[] shortArray) throws IOException {
        readShortArray(inputStream, shortArray);
        IntStream.range(0, shortArray.length).forEach(i -> data[i] = (int) shortArray[i]);
    }

    public static void readShort2IntLine(RandomAccessFile inputStream, int[] data) throws IOException {
        byte b;
        byte[] buffer = new byte[Sizeof.SHORT];
        int i = 0;
        int idx = 0;
        while ((b = inputStream.readByte()) != 10) {
            buffer[i++] = b;
            if (i == 1) {
                i = 0;
                short s = readShort(buffer);
                if (idx < data.length) {
                    data[idx] = s;
                }
                idx++;
            }
        }
        /**
         * padding

         */
        for (int j = idx; j < data.length; j++) {
            data[j] = 0;
        }
    }

    public static void readShort2IntLine(String txt, int[] data) throws IOException {
        String[] bs = txt.split("");
        byte[] buffer = new byte[Sizeof.SHORT];
        int idx = 0;
        for (int i = 0; i < bs.length; i++) {
            buffer[i % 2] = Byte.parseByte(bs[i]);
            if (i + 1 % 2 == 0) {
                short s = readShort(buffer);
                data[idx] = s;
                idx++;
            }
        }
        /**
         * padding

         */
        for (int j = idx; j < data.length; j++) {
            data[j] = 0;
        }
    }

    public static void writeFloat(RandomAccessFile outputStream, Tensor data) throws IOException {
        if (data.isHasGPU()) {
            data.syncHost();
        }
//        for (int i = 0; i < data.data.length; i++) {
//            writeFloat(outputStream, data.data[i]);
//        }

        writeFloatArray(outputStream, data.getData());
    }

    /**
     * 使用缓冲区批量写入
     *
     * @param outputStream
     * @param data
     * @throws IOException
     */
    public static void writeFloatArray(RandomAccessFile outputStream, float[] data) throws IOException {
        final int BUFFER_SIZE = 16384; // 16KB缓冲区
        byte[] buffer = new byte[BUFFER_SIZE];
        int bufferIndex = 0;

        for (float value : data) {
            int fbit = Float.floatToIntBits(value);

            // 直接写入缓冲区（小端序）
            buffer[bufferIndex++] = (byte) fbit ;
            buffer[bufferIndex++] = (byte) (fbit >> 8);
            buffer[bufferIndex++] = (byte) (fbit >> 16);
            buffer[bufferIndex++] = (byte) (fbit >> 24);

            if (bufferIndex >= BUFFER_SIZE) {
                outputStream.write(buffer, 0, bufferIndex);
                bufferIndex = 0;
            }
        }

        if (bufferIndex > 0) {
            outputStream.write(buffer, 0, bufferIndex);
        }
    }

    public static void writeInt(RandomAccessFile outputStream, int[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            writeInt(outputStream, data[i]);
        }
    }

    public static void writeFloat(RandomAccessFile outputStream, float val) throws IOException {
        byte[] buffer = float2byte(val);
        outputStream.write(buffer);
    }

    public static void writeInt(RandomAccessFile outputStream, int val) throws IOException {
        byte[] buffer = int2byte(val);
        outputStream.write(buffer);
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

    public static int readInt(RandomAccessFile inputStream) throws IOException {
        int retVal;
        byte[] buffer = new byte[Sizeof.INT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getInt();
        return retVal;
    }

    /**
     * unint16
     *
     * @param inputStream
     * @return
     * @throws IOException
     */
    public static short readShort(RandomAccessFile inputStream) throws IOException {
        byte[] buffer = new byte[Sizeof.SHORT];
        inputStream.readFully(buffer);
        return b2s(buffer);
    }

    /**
     * unint16
     *
     * @param inputStream
     * @return
     * @throws IOException
     */
    public static void readShortArray(RandomAccessFile inputStream, short[] cache) throws IOException {
        byte[] buffer = new byte[Sizeof.SHORT * cache.length];
        inputStream.readFully(buffer);
        convertByteArrayToShortArray(buffer, cache);
    }

    /**
     * unint16
     *
     * @param inputStream
     * @return
     * @throws IOException
     */
    public static short readShort(byte[] buffer) throws IOException {
        short retVal;
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getShort();
        return retVal;
    }

    /**
     * unint16
     *
     * @param inputStream
     * @return
     * @throws IOException
     */
    public static int readInt(byte[] buffer) throws IOException {
        int retVal;
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getInt();
        return retVal;
    }

    /**
     * 低字节
     *
     * @param i
     * @return
     */
    public static byte[] int2byte(int val) {
        byte[] b = new byte[4];
        for (int i = 0; i < 4; i++) {
            b[i] = (byte) (val >> (24 - i * 8));
        }
        // 翻转数组
        int len = b.length;
        // 建立一个与源数组元素类型相同的数组
        byte[] dest = new byte[len];
        // 为了防止修改源数组，将源数组拷贝一份副本
        System.arraycopy(b, 0, dest, 0, len);
        byte temp;
        // 将顺位第i个与倒数第i个交换
        for (int i = 0; i < len / 2; ++i) {
            temp = dest[i];
            dest[i] = dest[len - i - 1];
            dest[len - i - 1] = temp;
        }
        return dest;
    }

    public static float byte2int(byte[] b, int index) {
        int l;
        l = b[index + 0];
        l &= 0xff;
        l |= ((long) b[index + 1] << 8);
        l &= 0xffff;
        l |= ((long) b[index + 2] << 16);
        l &= 0xffffff;
        l |= ((long) b[index + 3] << 24);
        return l;
    }

    public static byte[] float2byte(float f) {
        // 把float转换为byte[]
        int fbit = Float.floatToIntBits(f);
        byte[] b = new byte[4];
        for (int i = 0; i < 4; i++) {
            b[i] = (byte) (fbit >> (24 - i * 8));
        }
        // 翻转数组
        int len = b.length;
        // 建立一个与源数组元素类型相同的数组
        byte[] dest = new byte[len];
        // 为了防止修改源数组，将源数组拷贝一份副本
        System.arraycopy(b, 0, dest, 0, len);
        byte temp;
        // 将顺位第i个与倒数第i个交换
        for (int i = 0; i < len / 2; ++i) {
            temp = dest[i];
            dest[i] = dest[len - i - 1];
            dest[len - i - 1] = temp;
        }
        return dest;
    }

    public static float byte2float(byte[] b, int index) {
        int l;
        l = b[index + 0];
        l &= 0xff;
        l |= ((long) b[index + 1] << 8);
        l &= 0xffff;
        l |= ((long) b[index + 2] << 16);
        l &= 0xffffff;
        l |= ((long) b[index + 3] << 24);
        return Float.intBitsToFloat(l);
    }

    //	public static byte[] short2byte(short s) {
    //		byte[] bytes = new byte[2];
    //		bytes[0] = (byte)(s >> 8);
    //		bytes[1] = (byte)s;
    //		return bytes;
    //	}
    //
    //	public static short byte2short(byte[] bytes) {
    //		return (short) ((bytes[0] & 0xff) | ((bytes[1] & 0xff) << 8));
    //	}
    public static byte[] s2b(short num) {
        ByteBuffer buffer = ByteBuffer.allocate(2);
        buffer.putShort(num);
        byte[] bytes = buffer.array();
        return bytes;
    }

    public static short b2s(byte[] b) {
        ByteBuffer byteBuffer2 = ByteBuffer.allocate(2);
        byteBuffer2.put(b);
        return byteBuffer2.getShort(0);
    }

    public static void convertByteArrayToShortArray(byte[] byteArray, short[] shortArray) {
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        buffer.order(ByteOrder.BIG_ENDIAN); // 根据需要选择字节序，这里选择大端模式
        buffer.asShortBuffer().get(shortArray); // 直接从buffer读取到short数组
    }

    public static byte[] short2byte(short s) {
        byte[] b = new byte[2];
        for (int i = 0; i < 2; i++) {
            int offset = 16 - (i + 1) * 8; //因为byte占4个字节，所以要计算偏移量
            b[i] = (byte) ((s >> offset) & 0xff); //把16位分为2个8位进行分别存储
        }
        return b;
    }

    public static short byte2short(byte[] b) {
        short l = 0;
        for (int i = 0; i < 2; i++) {
            l <<= 8; //<<=和我们的 +=是一样的，意思就是 l = l << 8
            l |= (b[i] & 0xff); //和上面也是一样的  l = l | (b[i]&0xff)
        }
        return l;
    }
}

