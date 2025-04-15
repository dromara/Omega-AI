package com.omega.engine.check;

import com.omega.common.tensor.Tensor;

public class VailCodeCheck extends BaseCheck {
    private static int labelClassLength = 4;

    @Override
    public float check(Tensor output, Tensor label, String[] labelSet, boolean showErrorLabel) {
        // TODO Auto-generated method stub
        output.syncHost();
        float trueCount = 0;
        for (int b = 0; b < output.number; b++) {
            float[] preData = output.getByNumber(b);
            float[] trueData = label.getByNumber(b);
            //			System.out.println(JsonUtils.toJson(trueData));
            String preLabel = this.getLabel(preData, labelSet);
            String trueLabel = this.getLabel(trueData, labelSet);
            if (preLabel.equals(trueLabel)) {
                trueCount = trueCount + 1;
            } else {
                if (showErrorLabel) {
                    System.out.println(preLabel + "=" + trueLabel + ":" + preLabel.equals(trueLabel));
                }
            }
        }
        return trueCount;
    }

    public String getLabel(float[] data, String[] labelSet) {
        //		System.out.println(JsonUtils.toJson(data));
        int size = labelSet.length;
        String label = "";
        for (int ls = 0; ls < labelClassLength; ls++) {
            float max = -3.402823466e+38F;
            int maxIndex = -1;
            for (int i = 0; i < size; i++) {
                float val = data[ls * size + i];
                if (val >= max) {
                    maxIndex = i;
                    max = val;
                }
            }
            label += labelSet[maxIndex];
        }
        return label;
    }
}

