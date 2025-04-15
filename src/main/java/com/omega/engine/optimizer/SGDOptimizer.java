package com.omega.engine.optimizer;

import com.omega.common.config.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.network.Network;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * Stochastic Gradient Descent
 *
 * @author Administrator
 */
public class SGDOptimizer extends Optimizer {
    public SGDOptimizer(Network network, int trainTime, float error, boolean warmUp) throws Exception {
        super(network, 1, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = 1;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
    }

    public SGDOptimizer(Network network, int trainTime, float error, LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
        super(network, 1, trainTime, error, warmUp);
        // TODO Auto-generated constructor stub
        this.batchSize = 1;
        this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
        this.learnRateUpdate = learnRateUpdate;
    }

    @Override
    public void train(BaseData trainingData) {
        // TODO Auto-generated method stub
        try {
            Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth());
            Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize);
            for (int i = 0; i < this.trainTime; i++) {
                this.trainIndex = i;
                if (this.currentError <= this.error && this.trainIndex >= this.minTrainTime) {
                    break;
                }
                this.loss.clear();
                this.lossDiff.clear();
                /**
                 * random data index

                 */
                int[] indexs = MathUtils.randomInt(trainingData.number - 1, this.batchSize);
                trainingData.getRandomData(indexs, input, label);
                /**
                 * forward

                 */
                Tensor output = this.network.forward(input);
                /**
                 * loss

                 */
                this.loss = this.network.loss(output, label);
                /**
                 * loss diff

                 */
                this.lossDiff = this.network.lossDiff(output, label);
                /**
                 * current time error

                 */
                this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
                /**
                 * update learning rate

                 */
                this.updateLR(this.lr_step);
                /**
                 * back

                 */
                this.network.back(this.lossDiff);
                float error = this.accuracy(output, label, trainingData.labelSet);
                System.out.println("training[" + this.trainIndex + "] accuracy:{" + error + "%} (lr:" + this.network.learnRate + ") currentError:" + this.currentError);
            }
            /**
             * 停止训练

             */
            System.out.println("training finish. [" + this.trainIndex + "] finalError:" + this.currentError);
            //			System.out.println(JsonUtils.toJson(this.network.layerList));
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

