% 2021/11/24 written by Fan Yifeng
% Homework 1: Jack’s Car Rental Problem

% P1(i,j), prob. of car number at 1st location change from i to j
P1=zeros(21);
for i=0:1:20
    for j=0:1:20
        pij=0;
        for k=0:1:i
            if j==20
                pij=pij+poisspdf(i-k,3)*poisscdf(j-k-1,3,'upper');
            else
                pij=pij+poisspdf(i-k,3)*poisspdf(j-k,3);
            end
        end
        if j==20
            pij=pij+poisscdf(i,3,'upper')*poisscdf(j-1,3,'upper');
        else
            pij=pij+poisscdf(i,3,'upper')*poisspdf(j,3);
        end
        P1(i+1,j+1)=pij;
    end 
end

% P2(i,j), prob. of car number at 2nd location change from i to j
P2=zeros(21);
for i=0:1:20
    for j=0:1:20
        pij=0;
        for k=0:1:i
            if j==20
                pij=pij+poisspdf(i-k,4)*poisscdf(j-k-1,2,'upper');
            else
                pij=pij+poisspdf(i-k,4)*poisspdf(j-k,2);
            end
        end
        if j==20
            pij=pij+poisscdf(i,4,'upper')*poisscdf(j-1,2,'upper');
        else
            pij=pij+poisscdf(i,4,'upper')*poisspdf(j,2);
        end
        P2(i+1,j+1)=pij;
    end 
end

% R1(i), expected rented cars at 1st location
R1=zeros(21,1);
for i=1:1:20
    for j=1:1:i
        R1(i+1)=R1(i+1)+j*poisspdf(j,3);
    end
    R1(i+1)=R1(i+1)+i*poisscdf(i,3,'upper');
end

% R2(i), expected rented cars at 2nd location
R2=zeros(21,1);
for i=1:1:20
    for j=1:1:i
        R2(i+1)=R2(i+1)+j*poisspdf(j,4);
    end
    R2(i+1)=R2(i+1)+i*poisscdf(i,4,'upper');
end

% Calculate P, state transition probability matrix
% Calculate R, expected revenue from rental, given the state after action
P=zeros(441);
R=zeros(441,1);
for st1=1:1:441
    c1=coordinate(st1);
    for st2=1:1:441
        c2=coordinate(st2);
        P(st1,st2)=P1(c1(1)+1,c2(1)+1)*P2(c1(2)+1,c2(2)+1);
    end
    R(st1)=10*(R1(c1(1)+1)+R2(c1(2)+1));
end

save P
save R

function x=coordinate(state)
x=[0 0];
x(1)=floor((state-1)/21);
x(2)=mod(state-1,21);
end