function plot_acc_astrobot_neighbors(TPR, TNR, BalAcc, n_of_neigh)

tpr6=mean(TPR(n_of_neigh==6));
tpr5=mean(TPR(n_of_neigh==5));
tpr4=mean(TPR(n_of_neigh==4));
tpr3=mean(TPR(n_of_neigh==3));
tpr2=mean(TPR(n_of_neigh==2));


tnr6=mean(TNR(n_of_neigh==6));
tnr5=mean(TNR(n_of_neigh==5));
tnr4=mean(TNR(n_of_neigh==4));
tnr3=mean(TNR(n_of_neigh==3));
tnr2=mean(TNR(n_of_neigh==2));

ba6=mean(BalAcc(n_of_neigh==6));
ba5=mean(BalAcc(n_of_neigh==5));
ba4=mean(BalAcc(n_of_neigh==4));
ba3=mean(BalAcc(n_of_neigh==3));
ba2=mean(BalAcc(n_of_neigh==2));


figure('Position', [10 10 2000 900]), 

if length(unique(n_of_neigh))==4
    X = categorical({' Astrobot with 6 neighbors','Astrobot with 5 neighbors','Astrobot with 4 neighbors','Astrobot with 3 neighbors'});
    X = reordercats(X,{'Astrobot with 6 neighbors','Astrobot with 5 neighbors','Astrobot with 4 neighbors','Astrobot with 3 neighbors'});
    v = [tpr6 tnr6 ba6; tpr5 tnr5 ba5; tpr4 tnr4 ba4; tpr3 tnr3 ba3];
else
    X = categorical({' Astrobot with 6 neighbors','Astrobot with 5 neighbors','Astrobot with 4 neighbors','Astrobot with 3 neighbors','Astrobot with 2 neighbors'});
    X = reordercats(X,{'Astrobot with 6 neighbors','Astrobot with 5 neighbors','Astrobot with 4 neighbors','Astrobot with 3 neighbors','Astrobot with 2 neighbors'});
    v = [tpr6 tnr6 ba6; tpr5 tnr5 ba5; tpr4 tnr4 ba4; tpr3 tnr3 ba3; tpr2 tnr2 ba2];
end

l = cell(1,3);
l{1}='TPR'; l{2}='TNR'; l{3}='Balanced Accuracy';    
h = bar(X,v);
grid on
ylabel('Accuracy [%]','FontSize',15);
ylim([0,100])
legend(h,l, 'FontSize', 20);
set(gca,'FontSize',15)


end

