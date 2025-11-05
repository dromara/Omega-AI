package com.omega.example.yolo.test.deepsort;

/**
 * 简单的卡尔曼滤波器
 * 用于目标追踪
 */
public class KalmanFilter {
    private double[] state; // [x, y, w, h, vx, vy, vw, vh]
    private double[][] errorCov;
    private double[][] transitionMatrix;
    private double[][] measurementMatrix;
    private double[][] processNoise;
    private double[][] measurementNoise;
    
    
    public KalmanFilter(Rect initialBbox) {
        // 8维状态向量: [x, y, w, h, vx, vy, vw, vh]
        state = new double[8];
        state[0] = initialBbox.x; // x
        state[1] = initialBbox.y; // y
        state[2] = initialBbox.width; // width
        state[3] = initialBbox.height; // height
        // 速度初始化为0
        for (int i = 4; i < 8; i++) {
            state[i] = 0.0;
        }
        
        // 初始化误差协方差矩阵
        errorCov = createIdentityMatrix(8);
        
        // 状态转移矩阵（匀速模型）
        transitionMatrix = createIdentityMatrix(8);
        for (int i = 0; i < 4; i++) {
            transitionMatrix[i][i + 4] = 1.0; // 位置 = 位置 + 速度
        }
        
        // 观测矩阵（只能观测到位置，不能观测到速度）
        measurementMatrix = new double[4][8];
        for (int i = 0; i < 4; i++) {
            measurementMatrix[i][i] = 1.0;
        }
        
        // 过程噪声和观测噪声
        processNoise = createIdentityMatrix(8);
        measurementNoise = createIdentityMatrix(4);
        
        // 调整噪声大小
        scaleMatrix(processNoise, 0.03);
        scaleMatrix(measurementNoise, 0.1);
    }
    
    public Rect predict() {
        // 状态预测: state = transitionMatrix * state
        double[] newState = matrixVectorMultiply(transitionMatrix, state);
        System.arraycopy(newState, 0, state, 0, state.length);
        
        // 误差协方差预测: errorCov = transitionMatrix * errorCov * transitionMatrix^T + processNoise
        double[][] temp1 = matrixMultiply(transitionMatrix, errorCov);
        double[][] transitionMatrixT = transposeMatrix(transitionMatrix);
        double[][] temp2 = matrixMultiply(temp1, transitionMatrixT);
        errorCov = matrixAdd(temp2, processNoise);
        
        // 返回预测的边界框
        int x = (int) Math.round(state[0]);
        int y = (int) Math.round(state[1]);
        int width = (int) Math.round(state[2]);
        int height = (int) Math.round(state[3]);
        
        return new Rect(x, y, width, height);
    }
    
    public void update(Rect measurement) {
        // 计算卡尔曼增益: K = errorCov * H^T * (H * errorCov * H^T + R)^(-1)
        double[][] measurementMatrixT = transposeMatrix(measurementMatrix);
        double[][] temp1 = matrixMultiply(errorCov, measurementMatrixT);
        double[][] temp2 = matrixMultiply(measurementMatrix, errorCov);
        double[][] temp3 = matrixMultiply(temp2, measurementMatrixT);
        double[][] s = matrixAdd(temp3, measurementNoise);
        double[][] sInv = invertMatrix(s);
        double[][] kalmanGain = matrixMultiply(temp1, sInv);
        
        // 状态更新
        double[] measurementVec = {measurement.x, measurement.y, measurement.width, measurement.height};
        double[] predictedMeasurement = matrixVectorMultiply(measurementMatrix, state);
        double[] innovation = vectorSubtract(measurementVec, predictedMeasurement);
        double[] correction = matrixVectorMultiply(kalmanGain, innovation);
        state = vectorAdd(state, correction);
        
        // 误差协方差更新
        double[][] identity = createIdentityMatrix(8);
        double[][] temp4 = matrixMultiply(kalmanGain, measurementMatrix);
        double[][] temp5 = matrixSubtract(identity, temp4);
        errorCov = matrixMultiply(temp5, errorCov);
    }
    
    // 工具方法：创建单位矩阵
    private double[][] createIdentityMatrix(int size) {
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            matrix[i][i] = 1.0;
        }
        return matrix;
    }
    
    // 工具方法：矩阵向量乘法
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    // 工具方法：矩阵乘法
    private double[][] matrixMultiply(double[][] a, double[][] b) {
        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;
        double[][] result = new double[aRows][bCols];
        
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0.0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    
    // 工具方法：矩阵加法
    private double[][] matrixAdd(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    
    // 工具方法：矩阵减法
    private double[][] matrixSubtract(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }
    
    // 工具方法：矩阵转置
    private double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    // 工具方法：向量减法
    private double[] vectorSubtract(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }
    
    // 工具方法：向量加法
    private double[] vectorAdd(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    // 工具方法：矩阵缩放
    private void scaleMatrix(double[][] matrix, double scalar) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] *= scalar;
            }
        }
    }
    
    // 工具方法：矩阵求逆（使用伴随矩阵法，适用于小矩阵）
    private double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] inverse = new double[n][n];
        
        // 计算行列式
        double det = determinant(matrix, n);
        
        if (Math.abs(det) < 1e-10) {
            // 行列式接近0，返回单位矩阵
            return createIdentityMatrix(n);
        }
        
        // 计算伴随矩阵
        double[][] adj = adjugate(matrix, n);
        
        // 计算逆矩阵: inverse = adjugate / determinant
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = adj[i][j] / det;
            }
        }
        
        return inverse;
    }
    
    // 计算矩阵的行列式
    private double determinant(double[][] matrix, int n) {
        if (n == 1) return matrix[0][0];
        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        
        double det = 0;
        int sign = 1;
        
        for (int i = 0; i < n; i++) {
            double[][] submatrix = getSubmatrix(matrix, 0, i, n);
            det += sign * matrix[0][i] * determinant(submatrix, n - 1);
            sign = -sign;
        }
        
        return det;
    }
    
    // 获取子矩阵（用于行列式计算）
    private double[][] getSubmatrix(double[][] matrix, int excludingRow, int excludingCol, int n) {
        double[][] submatrix = new double[n - 1][n - 1];
        int r = 0;
        for (int i = 0; i < n; i++) {
            if (i == excludingRow) continue;
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (j == excludingCol) continue;
                submatrix[r][c] = matrix[i][j];
                c++;
            }
            r++;
        }
        return submatrix;
    }
    
    // 计算伴随矩阵
    private double[][] adjugate(double[][] matrix, int n) {
        double[][] adj = new double[n][n];
        
        if (n == 1) {
            adj[0][0] = 1;
            return adj;
        }
        
        int sign = 1;
        double[][] temp = new double[n - 1][n - 1];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // 获取余子式
                getCofactor(matrix, temp, i, j, n);
                
                // 计算代数余子式
                sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj[j][i] = sign * determinant(temp, n - 1);
            }
        }
        
        return adj;
    }
    
    // 获取余子式
    private void getCofactor(double[][] matrix, double[][] temp, int p, int q, int n) {
        int i = 0, j = 0;
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (row != p && col != q) {
                    temp[i][j++] = matrix[row][col];
                    if (j == n - 1) {
                        j = 0;
                        i++;
                    }
                }
            }
        }
    }
    
    // 获取当前状态（用于调试）
    public double[] getState() {
        return state.clone();
    }
    
    // 使用示例
    public static void main(String[] args) {
        // 创建初始边界框
    	Rect initialBbox = new Rect(100, 100, 50, 50);
        KalmanFilter filter = new KalmanFilter(initialBbox);
        
        // 预测步骤
        Rect prediction = filter.predict();
        System.out.println("预测结果: " + prediction);
        
        // 更新步骤（使用新的测量值）
        Rect measurement = new Rect(105, 102, 48, 49);
        filter.update(measurement);
        
        // 再次预测
        prediction = filter.predict();
        System.out.println("更新后预测: " + prediction);
    }
}