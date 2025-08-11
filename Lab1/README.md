##Example Commands to Run##

python train_tf.py -m mlp -z 8 -e 20 -b 128 -s mlp_model.h5 -p mlp_loss.png

python main_tf.py -l mlp_model.h5 -m mlp -z 8

##Commands to Run as CNN##

python train_tf.py -m conv -e 20 -b 128 -s conv_model.h5 -p conv_loss.png

python main_tf.py -l conv_model.h5 -m conv

