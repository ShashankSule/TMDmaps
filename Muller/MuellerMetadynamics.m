% function samples = MuellerMetadynamics()
close all
%% parameters for metadynamics
Ndeposit = 1000;
Nbumps = 1000;
height = 5;
sig = 0.05;
sig2 = sig^2;
%% graphics
N = 100;
XMIN = -1.5; XMAX = 1.5;
YMIN = -0.5; YMAX = 2;
t1 = linspace(XMIN,XMAX,N);
t2 = linspace(YMIN,YMAX,N);
[x, y] = meshgrid(t1, t2);
V = mueller([x(:),y(:)]);
% figure;
% contour(x,y,reshape(V,N,N),-200:10:400,'linewidth',1,'color','k')
% hold on
% grid;
%% parameters for time stepping
h = 1e-4; % time step
Temp = 20; % temperature
sqh = sqrt(h*2*Temp);
traj = zeros(Ndeposit+1,2);
subsample = 1e3;
samples = zeros(floor(Ndeposit*Nbumps/subsample),2); 
x0 = [0,0];
coef = zeros(Nbumps,1);
xbump = zeros(Nbumps,2);
i=1; 
for k = 1 : Nbumps
    traj = zeros(Ndeposit+1,2);
    traj(1,:) = x0;    
    w = sqh*randn(Ndeposit,2);
    for j = 1 : Ndeposit
        dVmueller = grad_mueller(traj(j,:));
        aux1 = traj(j,1)-xbump(:,1);
        aux2 = traj(j,2)-xbump(:,2);      
        dVbump = sum((coef*[1,1]).*[aux1,aux2].*(exp(-0.5*(aux1.^2+aux2.^2)/sig2)*[1,1]),1)/sig2;
        traj(j+1,:) = traj(j,:) - h*(dVmueller-dVbump) + w(j,:);
        % subsampling routine 
        if mod((k-1)*Ndeposit + j, subsample) == 0 
            samples(i,:) = traj(j,:);
            i = i+1; 
        end
    end
    % plot(traj(:,1),traj(:,2));
    % drawnow;
    % fprintf('k = %d\n',k);
    % deposit bump
    x0 = traj(j+1,:);
    xbump(k,:) = x0;
    coef(k) = height;
end
 
%% Sample the modified potential 
figure;
contour(x,y,reshape(V,N,N),-200:10:400,'linewidth',1,'color','k')
hold on
grid;
Ndeposit = 1e6; 
samples = zeros(1e4, 2); 
traj = zeros(Ndeposit+1,2);
traj(1,:) = x0;    
w = sqh*randn(Ndeposit,2);
i=1; 
for j = 1 : Ndeposit
    dVmueller = grad_mueller(traj(j,:));
    aux1 = traj(j,1)-xbump(:,1);
    aux2 = traj(j,2)-xbump(:,2);      
    dVbump = sum((coef*[1,1]).*[aux1,aux2].*(exp(-0.5*(aux1.^2+aux2.^2)/sig2)*[1,1]),1)/sig2;
    traj(j+1,:) = traj(j,:) - h*(dVmueller-dVbump) + w(j,:);
    % subsampling routine 
    if mod(j, 100) == 0 
        samples(i,:) = traj(j,:);
        i = i+1; 
    end
%     plot(traj(1:j,1),traj(1:j,2), 'bo')
%     drawnow;
end
plot(samples(1:j,1),samples(1:j,2), 'bo')
drawnow;
save('Muller_Data_Metadynamics_longsample_20.mat', 'samples');
% end

%%
function V = mueller(x)
a = [-1,-1,-6.5,0.7];
b = [0,0,11,0.6];
c = [-10,-10,-6.5,0.7];
D = [-200,-100,-170,15];
X = [1,0,-0.5,-1];
Y = [0,0.5,1.5,1];
[t,~]=size(x);
V=zeros(t,1);
for i = 1 : 4
    Vnew = D(i)*exp(a(i)*(x(:,1)-X(i)).^2+b(i)*(x(:,1)-X(i)).*(x(:,2)-Y(i))+...
        c(i)*(x(:,2)-Y(i)).^2);
    V=V+Vnew;
end
        
end

%%
function dV = grad_mueller(x)
a = [-1,-1,-6.5,0.7];
b = [0,0,11,0.6];
c = [-10,-10,-6.5,0.7];
D = [-200,-100,-170,15];
X = [1,0,-0.5,-1];
Y = [0,0.5,1.5,1];
[t,~]=size(x);
dV=zeros(t,2);
for i = 1 : 4
    Vnew = D(i)*exp(a(i)*(x(:,1)-X(i)).^2+b(i)*(x(:,1)-X(i)).*(x(:,2)-Y(i))+...
        c(i)*(x(:,2)-Y(i)).^2);
    dVnew(:,1)=(2*a(i)*(x(:,1)-X(i))+b(i)*(x(:,2)-Y(i))).*Vnew;
    dVnew(:,2)=(2*c(i)*(x(:,2)-Y(i))+b(i)*(x(:,1)-X(i))).*Vnew;
    dV=dV+dVnew;
end
        
end