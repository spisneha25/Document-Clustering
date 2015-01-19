function [] = mainfunc( )

load('data');
T = X(:,1:100);
label = X(:,101);

IDX = mycluster(T,4);

acc=AccMeasure(label,IDX)

end
