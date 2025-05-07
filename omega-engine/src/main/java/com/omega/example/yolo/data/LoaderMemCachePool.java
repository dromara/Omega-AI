package com.omega.example.yolo.data;

import java.util.ArrayList;
import java.util.List;

/**
 * loader memery cache pool
 *
 * @author Administrator
 */
public class LoaderMemCachePool {
    private List<MemeryBlock> blocks = new ArrayList<MemeryBlock>();

    public synchronized MemeryBlock getBlock(int size) {
        //		System.out.println(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            MemeryBlock block = blocks.get(i);
            if (block.getSize() == size && !block.isStatus()) {
                block.setStatus(true);
                return block;
            }
        }
        MemeryBlock block = new MemeryBlock(size);
        block.setStatus(true);
        blocks.add(block);
        return block;
    }

    public void free(MemeryBlock block) {
        block.setStatus(false);
    }

    public void free(float[] data) {
        for (int i = 0; i < blocks.size(); i++) {
            MemeryBlock block = blocks.get(i);
            if (block.getData() == data) {
                block.setStatus(false);
                break;
            }
        }
    }
}

