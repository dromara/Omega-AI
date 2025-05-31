package com.omega.engine.optimizer;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.tensor.Tensor;

/**
 * Batch Gradient Descent
 *
 * @author Administrator
 */
public class BGDOptimizer extends Optimizer {
    public BGDOptimizer(Network network, int batchSize, int trainTime, float error, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
    }

    public BGDOptimizer(Network network, int batchSize, int trainTime, float error, LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
        super(network, batchSize, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = batchSize;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.learnRateUpdate = learnRateUpdate;
    }

    @Override
    public void train(BaseData trainingData) {
        // TODO Auto-generated method stub
        try {
            for (int i = 0; i < this.trainTime; i++) {
                if (this.currentError <= this.error && this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.loss.clear();
                this.lossDiff.clear();
                /**
                 * forward

                 */
                Tensor output = this.network.forward(trainingData.input);
                /**
                 * loss

                 */
                this.loss = this.network.loss(output, trainingData.label);
                /**
                 * lossDiff

                 */
                this.lossDiff = this.network.lossDiff(output, trainingData.label);
                /**
                 * current time error

                 */
                this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                /**
                 * back

                 */
                this.network.back(this.lossDiff);
                this.trainIndex = i;
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
            System.out.println(JsonUtils.toJson(this.network.layerList));
        } catch (Exception e) {
            // TODO: handle exception
            e.printStackTrace();
        }
    }

    @Override
    public void train(BaseData trainingData, BaseData testData) {
        // TODO Auto-generated method stub
    }

    @Override
    public void train(BaseData trainingData, BaseData validata, BaseData testData) {
        // TODO Auto-generated method stub
    }
}

