function [Ones,Zeros] = count_zeros_and_ones(Data)
%This function count the total number of astrobots which converge to their target position and the
%total number of astrobots which don't converge to their target position

n_of_sim=size(Data,3);
astrobots=size(Data,2);

Ones=0;
Zeros=0;

for i=1:n_of_sim
    
    Ones = Ones + sum(Data(5,:,i)==ones(1,astrobots));
    Zeros = Zeros + sum(Data(5,:,i)==zeros(1,astrobots));
    
end

str1=['The total number of astrobots that converge is: ' num2str(Ones)];
str2=['The total number of astrobots that do not converge is: ' num2str(Zeros)];
disp(str1)
disp(str2)

end

