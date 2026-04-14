package com.omega.example.transformer.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;

public class JsonFileIterator implements Iterator<Map.Entry<String, Object>>, AutoCloseable {

    private final JsonReader reader;
    private final int bufferSize;
    
    private Map.Entry<String, Object> nextEntry;
    private boolean finished = false;
    private boolean closed = false;

    public JsonFileIterator(String path) throws IOException {
        this(path, 64 * 1024); // 默认 64KB 缓冲
    }

    public JsonFileIterator(String path, int bufferSize) throws IOException {
        this.bufferSize = bufferSize;
        FileInputStream fis = new FileInputStream(path);
        InputStreamReader isr = new InputStreamReader(fis, StandardCharsets.UTF_8);
        BufferedReader br = new BufferedReader(isr, bufferSize);
        this.reader = new JsonReader(br);
        this.reader.setLenient(true); // 兼容非严格 JSON
        
        // 定位到顶层对象开始
        if (reader.peek() == JsonToken.BEGIN_OBJECT) {
            reader.beginObject();
        } else {
            throw new IOException("Expected JSON object at root level, but found: " + reader.peek());
        }
        prefetch();
    }

    /**
     * 预取下一个 entry，供 hasNext/next 使用
     */
    private void prefetch() throws IOException {
        if (closed) return;
        
        if (reader.hasNext()) {
            String name = reader.nextName();
            Object value = readJsonValue(reader);  // 流式解析 value
            nextEntry = new AbstractMap.SimpleImmutableEntry<>(name, value);
        } else {
            finished = true;
            close();  // 自动关闭资源
        }
    }

    @Override
    public boolean hasNext() {
        return !finished;
    }

    @Override
    public Map.Entry<String, Object> next() {
        if (!hasNext()) {
            throw new NoSuchElementException("No more entries in JSON file");
        }
        Map.Entry<String, Object> result = nextEntry;
        try {
            prefetch();  // 预取下一个，当前 entry 可被 GC
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read next JSON entry", e);
        }
        return result;
    }

    @Override
    public void close() {
        if (closed) return;
        try {
            reader.close();
        } catch (IOException ignored) {}
        closed = true;
        finished = true;
        nextEntry = null;
    }

    /**
     * 流式解析任意 JSON value（object/array/primitive）
     * ⚠️ 注意：嵌套结构会完整加载到内存，请确保单个 value 大小可控
     */
    @SuppressWarnings("unchecked")
    private Object readJsonValue(JsonReader reader) throws IOException {
        switch (reader.peek()) {
            case BEGIN_OBJECT:
                return readJsonObject(reader);
            case BEGIN_ARRAY:
                return readJsonArray(reader);
            case STRING:
                return reader.nextString();
            case NUMBER:
                return reader.nextDouble(); // 如需精度可用 nextString() + BigDecimal
            case BOOLEAN:
                return reader.nextBoolean();
            case NULL:
                reader.nextNull();
                return null;
            default:
                throw new IllegalStateException("Unexpected token: " + reader.peek());
        }
    }

    private Map<String, Object> readJsonObject(JsonReader reader) throws IOException {
        Map<String, Object> map = new LinkedHashMap<>();
        reader.beginObject();
        while (reader.hasNext()) {
            String name = reader.nextName();
            map.put(name, readJsonValue(reader));
        }
        reader.endObject();
        return map;
    }

    private List<Object> readJsonArray(JsonReader reader) throws IOException {
        List<Object> list = new ArrayList<>();
        reader.beginArray();
        while (reader.hasNext()) {
            list.add(readJsonValue(reader));
        }
        reader.endArray();
        return list;
    }

    /**
     * 转换为 Java 8 Stream（自动管理资源）
     * 使用示例：
     * <pre>
     * try (Stream<Map.Entry<String, Object>> stream = JsonFileIterator.streamOf("data.json")) {
     *     stream.filter(e -> e.getKey().startsWith("user"))
     *           .forEach(e -> process(e.getValue()));
     * }
     * </pre>
     */
    public static Stream<Map.Entry<String, Object>> streamOf(String path) throws IOException {
        JsonFileIterator iterator = new JsonFileIterator(path);
        // Spliterator 特性：有序、非并行、值不可变
        Spliterator<Map.Entry<String, Object>> spliterator = Spliterators.spliteratorUnknownSize(
                iterator, Spliterator.ORDERED | Spliterator.NONNULL | Spliterator.IMMUTABLE);
        return StreamSupport.stream(spliterator, false)
                .onClose(iterator::close);  // 关键：流关闭时释放文件句柄
    }
}