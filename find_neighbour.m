function index_neigh = find_neighbour(Data, astrobots_idx)

%This function return the indices of the neighbours of a specific astrobot
%in the focal plane

%Maximum allowed distance between astrobots in the focal plane is L
L=36;

Data_single_positioner=Data(:,astrobots_idx,1);
x_single=Data_single_positioner(1);
y_single=Data_single_positioner(2);
index_neigh=[];

for i=1:size(Data,2)
   if (i==astrobots_idx) 
       continue
   end
   
   if distance_plane(x_single,y_single,Data(1,i,1),Data(2,i,1)) < L
       index_neigh=[index_neigh,i];
   end
end




end

