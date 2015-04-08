require 'cunn';
holder=torch.load('model_bigboy_fold_1_epoch_29.net')
epoch_29=holder:getParameters()

holder=torch.load('model_bigboy_fold_1_epoch_28.net')
epoch_28=holder:getParameters()


x=epoch_29-epoch_28



xx=x:reshape(75611,5)
x_epoch_29=epoch_29:reshape(75611,5)

x_epoch_28=epoch_28:reshape(75611,5)

epoch_28
