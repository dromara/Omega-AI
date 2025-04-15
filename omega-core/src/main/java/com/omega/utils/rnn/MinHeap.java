package com.omega.utils.rnn;

public class MinHeap {
    // 堆的存储结构 - 数组
    private float[] data;

    // 将一个数组传入构造方法，并转换成一个小根堆
    public MinHeap(float[] data) {
        this.data = data;
        buildHeap();
    }

    // 将数组转换成最小堆
    private void buildHeap() {
        // 完全二叉树只有数组下标小于或等于 (data.length) / 2 - 1 的元素有孩子结点，遍历这些结点。
        // *比如上面的图中，数组有10个元素， (data.length) / 2 - 1的值为4，a[4]有孩子结点，但a[5]没有*
        for (int i = (data.length) / 2 - 1; i >= 0; i--) {
            // 对有孩子结点的元素heapify
            heapify(i);
        }
    }

    private void heapify(int i) {
        // 获取左右结点的数组下标
        int l = left(i);
        int r = right(i);
        // 这是一个临时变量，表示 跟结点、左结点、右结点中最小的值的结点的下标
        int smallest = i;
        // 存在左结点，且左结点的值小于根结点的值
        if (l < data.length && data[l] < data[i])
            smallest = l;
        // 存在右结点，且右结点的值小于以上比较的较小值
        if (r < data.length && data[r] < data[smallest])
            smallest = r;
        // 左右结点的值都大于根节点，直接return，不做任何操作
        if (i == smallest)
            return;
        // 交换根节点和左右结点中最小的那个值，把根节点的值替换下去
        swap(i, smallest);
        // 由于替换后左右子树会被影响，所以要对受影响的子树再进行heapify
        heapify(smallest);
    }

    // 获取右结点的数组下标
    private int right(int i) {
        return (i + 1) << 1;
    }

    // 获取左结点的数组下标
    private int left(int i) {
        return ((i + 1) << 1) - 1;
    }

    // 交换元素位置
    private void swap(int i, int j) {
        float tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }

    // 获取对中的最小的元素，根元素
    public float getRoot() {
        return data[0];
    }

    // 替换根元素，并重新heapify
    public void setRoot(float root) {
        data[0] = root;
        heapify(0);
    }
}

