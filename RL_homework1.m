% 2021/11/24 written by Fan Yifeng
% Homework 1: Jack’s Car Rental Problem

% Run RL_homework1_data.m first
% Go to line 47-48 to switch between PI and VI

load P;
load R;

gamma=0.9;          % discount factor
epsilon=0.0001;     % accuracy in iterative policy evaluation
theta=1;            % accuracy in value iteration

vpi=zeros(441,1);   % initial value function

% Calculate Rsa, reward given state and action
Rsa=zeros(441,11);
for st=1:1:441
    for act=-5:1:5
        stm=transfer(st,act);
        Rsa(st,act+6)=R(stm)-2*abs(act);
    end
end

% Initial policy, can be chosen arbitrarily
% Random policy and static policy provided as example, uncomment to choose

% % Calculate initial policy (random move)
% PI=zeros(441,11);
% for st=1:1:441
%     [lower_bound,upper_bound]=bound(st);
%     prob=1/(upper_bound-lower_bound+1);
%     for act=lower_bound:1:upper_bound
%         PI(st,act+6)=prob;
%     end
% end

% Calculate initial policy (do not move)
PI=zeros(441,11);
for st=1:1:441
    PI(st,6)=1;
end

% main
% uncomment to choose between policy iteration and value iteration

Policy_Iteration(vpi,PI,Rsa,P,gamma,epsilon);
% Value_Iteration(vpi,PI,Rsa,P,gamma,theta);



function Policy_Iteration(vpi,PI,Rsa,P,gamma,epsilon)
policy_stable=0;
iter=0;
while policy_stable==0
    iter=iter+1;
    Rpi=UpdateRpi(PI,Rsa);
    Ppi=UpdatePpi(PI,P);
    vpi=IterativePolicyEvaluation(vpi,Rpi,Ppi,gamma,epsilon);
    PI_old=PI;
    PI=UpdatePI(vpi,Rsa,P,gamma);
    diff=sum(abs(PI-PI_old),'all')/2;
    fprintf('Iteration %d, policy change:', iter);
    disp(diff);
    plotpi(PI);
    plotv(vpi);
    drawnow
    if diff==0
        policy_stable=1;
    end
end
fprintf('Stable policy attained in %d iterations:\n\n',iter);
printpi(PI);
end

function Value_Iteration(vpi,PI,Rsa,P,gamma,theta)
iter=0;
vpi_diff=theta+100;
while vpi_diff>theta
    iter=iter+1;
    Rpi=UpdateRpi(PI,Rsa);
    Ppi=UpdatePpi(PI,P);
    vpi_old=vpi;
    vpi=IterativePolicyEvaluation_k(vpi,Rpi,Ppi,gamma,1);
    PI=UpdatePI(vpi,Rsa,P,gamma);
    vpi_diff=max(abs(vpi-vpi_old));
    fprintf('Iteration %d, value change:', iter);
    disp(vpi_diff);
    plotpi(PI);
    plotv(vpi);
    drawnow
end
fprintf('Optimal policy attained in %d iterations:\n\n',iter);
printpi(PI);
end

function Rpi=UpdateRpi(PI,Rsa)
Rpi=zeros(441,1);
for st=1:1:441
    Rpi(st)=PI(st,:)*Rsa(st,:)';
end
end

function Ppi=UpdatePpi(PI,P)
Ppi=zeros(441,441);
for st1=1:1:441
    [lower_bound,upper_bound]=bound(st1);
    for st2=1:1:441
        for act=lower_bound:1:upper_bound
            stm=transfer(st1,act);
            Ppi(st1,st2)=Ppi(st1,st2)+PI(st1,act+6)*P(stm,st2);
        end
    end
end
end

function PI=UpdatePI(vpi,Rsa,P,gamma)
PI=zeros(441,11);
vnext=P*vpi;
for st1=1:1:441
    [lower_bound,upper_bound]=bound(st1);
    best_act=0;
    max_value=-10000;
    for act=lower_bound:1:upper_bound
        stm=transfer(st1,act);
        value=Rsa(st1,act+6)+gamma*vnext(stm);
        if value>max_value
            best_act=act;
            max_value=value;
        end
    end
    PI(st1,best_act+6)=1;
end
end

% Iterative Policy Evaluation, ending when error < epsilon
function v_star=IterativePolicyEvaluation(v,Rpi,Ppi,gamma,epsilon)
error=epsilon+1;
while (error>epsilon)
    v_last=v;
    v=Rpi+gamma*Ppi*v;
    error=max(abs(v-v_last));
    % disp(error);
end
v_star=v;
end

% Iterative Policy Evaluation, ending after k iterations
function v_star=IterativePolicyEvaluation_k(v,Rpi,Ppi,gamma,k)
for i=1:1:k
    v=Rpi+gamma*Ppi*v;
end
v_star=v;
end

% In place realization of Iterative Policy Evaluation
function v_star=IterativePolicyEvaluation_InPlace(v,Rpi,Ppi,gamma,epsilon)
error=epsilon+1;
while (error>epsilon)
    v_last=v;
    for st=1:1:441
        v(st)=Rpi(st)+gamma*Ppi(st,:)*v;
    end
    error=max(abs(v-v_last));
    % disp(error);
end
v_star=v;
end

% Given initial state and action, return new state
function x=transfer(st,act)
x=st-20*act;
if islegal(st,act)==0
    x=st;
end
end

% Return whether action at a state is legal
function b=islegal(st,act)
b=1;
c=coordinate(st);
if (c(1)<act)||(c(2)<-act)||(c(2)+act>20)||(c(1)-act>20)
    b=0;
end
end

function x=stateindex(coord)
x=21*coord(1)+coord(2)+1;
end

function x=coordinate(state)
x=[0 0];
x(1)=floor((state-1)/21);
x(2)=mod(state-1,21);
end

% Return the bound of legal actions at a state
function [l,u]=bound(state)
c=coordinate(state);
l=-min([5 20-c(1) c(2)]);
u=min([5 c(1) 20-c(2)]);
end

function surfv(v)
V=reshape(v,[21 21]);
x=0:1:20;
y=0:1:20;
surf(x,y,V);
colorbar;
xlabel('#Cars at first location');
ylabel('#Cars at second location');
end

function plotv(v)
V=reshape(v,[21 21]);
subplot(1,2,2);
x=[0 20];
y=[0 20];
imagesc(x,y,V');
set(gca,'YDir','normal');
axis square;
colorbar;
xlabel('#Cars at second location') 
ylabel('#Cars at first location')
end

function plotpi(PI)
pi_matrix=zeros(21,21);
for i=1:1:441
    c=coordinate(i);
    for j=1:1:11
        if PI(i,j)==1
            pi_matrix(c(1)+1,c(2)+1)=j-6;
            break;
        end
    end
end
subplot(1,2,1);
x=[0 20];
y=[0 20];
clims=[-5 5];
imagesc(x,y,pi_matrix,clims);
set(gca,'YDir','normal');
axis square;
colorbar;
xlabel('#Cars at second location') 
ylabel('#Cars at first location')
end

function printpi(PI)
pi_matrix=zeros(21,21);
for i=1:1:441
    c=coordinate(i);
    for j=1:1:11
        if PI(i,j)==1
            pi_matrix(c(1)+1,c(2)+1)=j-6;
            break;
        end
    end
end
disp(pi_matrix);
end