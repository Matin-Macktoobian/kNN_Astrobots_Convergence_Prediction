function [output_conv_info, index_not_full_conv] = count_conv(Data)

%This function count the number of full convergence configuration and the
%number of not full convergence configuration

n_of_sim=size(Data,3);
astrobots=size(Data,2);

full_conv_conf=0;
not_full_conv_conf=0;
index_not_full_conv=[];
for i=1:n_of_sim
    
    if sum((Data(end,:,i)==ones(1,astrobots))) == astrobots
        full_conv_conf = full_conv_conf + 1;
    else
        not_full_conv_conf = not_full_conv_conf + 1;
        index_not_full_conv = [index_not_full_conv; i];
    end
end

output_conv_info=[full_conv_conf, not_full_conv_conf];

str1=['The number of full convergence configurations is:' num2str(full_conv_conf)];
str2=['The number of NOT full convergence configurations is:' num2str(not_full_conv_conf)];

disp(str1)
disp(str2)

end

