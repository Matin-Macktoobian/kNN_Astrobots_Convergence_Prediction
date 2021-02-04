function prob = prediction_probability_weighted_tot(Data,weight_ratio,which_astrobots)
switch nargin
        case 2
            n=size(Data,3);
            W=[]; predicted_output=[];
            for i=1:n
               W=[W;weight_ratio.*(Data(end,:,i)==zeros(1,size(Data,2))) + (Data(end,:,i)==ones(1,size(Data,2)))]; 
            end
            predicted_output=Data(5,:,:);
            predicted_output=permute(predicted_output,[3,2,1]);
            prob=sum(predicted_output,1)./sum(W,1);
        case 3
            n=size(Data,3);
            W=[]; predicted_output=[];
            for i=1:n
               W=[W;weight_ratio.*(Data(end,:,i)==zeros(1,size(Data,2))) + (Data(end,:,i)==ones(1,size(Data,2)))]; 
            end
            predicted_output=Data(5,:,:);
            predicted_output=permute(predicted_output,[3,2,1]);
            prob=sum(predicted_output,1)./sum(W,1);
            
            %Giving weight on the basis of the neighbour considered (which astrobots)
            weight=0*ones(1,size(Data,2));
            weight(which_astrobots)=1;
            prob=prob.*weight;
end

