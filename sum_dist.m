function sum = sum_dist(Start_conf,Target_conf)
%Start_conf is a 2 x M vector
%Target_conf is a 2 x M vector (2 because we need x and y)

M=size(Start_conf,2);

if size(Start_conf,2)~=size(Target_conf,2)
    msg='Starting configuration and target configuration have different size';
    error(msg);
end

sum=0;
for i=1:M
    sum=sum + distance_plane(Start_conf(1,i),Start_conf(2,i), Target_conf(1,i), Target_conf(2,i));
end
    

end

