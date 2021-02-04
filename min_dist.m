function [mindist, index] = min_dist(All_data,single_conf,n_neighb)
%This function returns the indices of the K closest configurations and the
%values of the distances.

n=size(All_data,3);
mindist=inf;
mindist_vec=[];
mindist_vec_plus_std=[];
for i=1:n
    
    %SUM DISTANCE
    d=sum_dist(All_data([3:4],:,i),single_conf([3:4],:));
    mindist_vec=[mindist_vec;d];
    
end

%Sorting for MEAN and SUM DISTANCE
[sorted_values, idx]=sort(mindist_vec,'ascend');
mindist=sorted_values(1:n_neighb);
index=idx(1:n_neighb);


end

