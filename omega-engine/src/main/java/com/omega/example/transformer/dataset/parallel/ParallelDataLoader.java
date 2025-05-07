package com.omega.example.transformer.dataset.parallel;

import com.omega.example.transformer.dataset.DatasetLoader;
import com.omega.example.transformer.dataset.SFTBinDataset;

import java.util.HashMap;
import java.util.Map;

public class ParallelDataLoader {
    private int count_it;
    private int[] rankIds;
    private DatasetLoader dataloader;
    private Map<Integer, ThreadDataset> dataloaders = new HashMap<Integer, ThreadDataset>();

    public ParallelDataLoader(DatasetLoader dataloader, int[] rankIds) {
        this.dataloader = dataloader;
        this.rankIds = rankIds;
        createThreads();
    }

    public void createThreads() {
        if (dataloader instanceof SFTBinDataset) {
            createSFTBinThreads();
        }
    }

    public void createSFTBinThreads() {
        SFTBinDataset dl = (SFTBinDataset) dataloader;
        int partSize = dl.number / rankIds.length;
        this.count_it = partSize / dl.getBatchSize();
        for (int i = 0; i < rankIds.length; i++) {
            SFTBinTheads thread = new SFTBinTheads(dl.getDataPath(), dl.max_len, dl.getBatchSize(), rankIds[i], partSize, dl.tokenizer, dl.getDataType());
            getDataloaders().put(rankIds[i], thread);
        }
    }

    public Map<Integer, ThreadDataset> getDataloaders() {
        return dataloaders;
    }

    public int getCount_it() {
        return count_it;
    }

    public void setCount_it(int count_it) {
        this.count_it = count_it;
    }
}

