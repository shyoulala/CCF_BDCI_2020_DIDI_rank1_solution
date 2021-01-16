tar -zxvf ../../data/raw_data/traffic-fix.tar.gz -C  ../../data/user_data/

cd ../../data/code/nn
#nn prepare
python data_prepare.py

#nn训练和预测
python nn_train_and_test_predict.py

#lgb 训练和预测
cd ../
python lgb_model.py

#融合
python merge_prob.py

