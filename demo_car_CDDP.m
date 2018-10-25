function demo_car_CDDP
% A demo of iLQG/DDP with car-parking dynamics
clc;
close all

fprintf(['\nA demonstration of the iLQG algorithm '...
'with car parking dynamics.\n'...
'for details see\nTassa, Mansard & Todorov, ICRA 2014\n'...
'\"Control-Limited Differential Dynamic Programming\"\n'])

% Set full_DDP=true to compute 2nd order derivatives of the 
% dynamics. This will make iterations more expensive, but 
% final convergence will be much faster (quadratic)
full_DDP = false;

% set up the optimization problem
DYNCST  = @(x,u,i) car_dyn_cst(x,u,full_DDP);
T       = 500;              % horizon
x0      = [1;1;pi*3/2;0];   % initial statesa
u0      = .1*randn(2,T);    % initial control input 
Op.plot = 0;               % plot the derivatives as well

% % prepare the visualization window and graphics callback
figure(9);
set(gcf,'name','car parking','Menu','none','NumberT','off')
set(gca,'xlim',[-8 8],'ylim',[-8 8],'DataAspectRatio',[1 1 1])
grid on
box on

% plot target configuration with light colors
handles = car_plot(x0', [0 0]');
fcolor  = get(handles,'facecolor');
ecolor  = get(handles,'edgecolor');
fcolor  = cellfun(@(x) (x+3)/4,fcolor,'UniformOutput',false);
ecolor  = cellfun(@(x) (x+3)/4,ecolor,'UniformOutput',false);
set(handles, {'facecolor','edgecolor'}, [fcolor ecolor])

% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','b','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;
% 

% === run the optimization!
[x,u]= CDDP(DYNCST, x0, u0, Op); %DYNCST

% animate the resulting trajectory
figure(9)
handles = [];
for i=1:T
   set(0,'currentfigure',9);
   delete(handles)
   handles = car_plot(x(:,i), u(:,i));
   drawnow    
end

function y = car_dynamics(x,u)

% === states and controls:
% x = [x y t v]' = [x; y; car_angle; front_wheel_velocity]
% u = [w a]'     = [front_wheel_angle; acceleration]

% constants
d  = 2.0;      % d = distance between back and front axles
h  = 0.03;     % h = timestep (seconds)

% controls
w  = u(1,:,:); % w = front wheel angle
a  = u(2,:,:); % a = front wheel acceleration

o  = x(3,:,:); % o = car angle
               % z = unit_vector(o)
z  = [cos(o); sin(o)]; 

v  = x(4,:,:); % v = front wheel velocity
f  = h*v;      % f = front wheel rolling distance
               % b = back wheel rolling distance
b  = d + f.*cos(w) - sqrt(d^2 - (f.*sin(w)).^2);
               % do = change in car angle
do = asin(sin(w).*f/d);

dy = [tt(b, z); do; h*a];   % change in state
y  = x + dy;                % new state


function c = car_cost(x, u)
% cost function for car-parking problem
% sum of 3 terms:
% lu: quadratic cost on controls
% lf: final cost on distance from target parking configuration
% lx: running cost on distance from origin to encourage tight turns

final = isnan(u(1,:));
u(:,final)  = 0;

cu  = 1e-2*[1 .01];         % control cost coefficients

cf  = [ .1  .1   1  .3]; 
pf  = [.01 .01 .01  .01]';    % smoothness scales for final cost

xd = [0 0 0 0]';

cx  = 1e-3*[0.8  0.8];          % running cost coefficients
px  = [.05 .05]';             % smoothness scales for running cost

% control cost
lu    = cu*u.^2;

% final cost
if any(final)
   llf      = cf*sabs(x(:,final)-xd,pf);
   lf       = double(final);
   lf(final)= llf;
else
   lf    = 0;
end


% lx2= 0.001/d(1,:);

lx = cx*sabs(x(1:2,:),px);

% total cost
c     =  lu + lx + lf;

function d = car_constraint(x, u)
% constraint fucntion for car driving
% control constraint (-0.5 0.5) wheel angle, (-2 2) acceleration
% state constraint (x+1)^2 + (y+3)^2 =1, x^2 + (y-3)^2=1;
d1 = -0.5 - u(1,:); % lb for wheel constraint
d2 = u(1,:) - 0.5; % ub for wheel constraint
d3 = -2 - u(2,:); % lb for acceleration
d4 = u(2,:) - 2; % ub for acc

% constants
d  = 2.0;      % d = distance between back and front axles
h  = 0.03;     % h = timestep (seconds)

% controls
w  = u(1,:,:); % w = front wheel angle
a  = u(2,:,:); % a = front wheel acceleration

o  = x(3,:,:); % o = car angle
               % z = unit_vector(o)
z  = [cos(o); sin(o)]; 

v  = x(4,:,:); % v = front wheel velocity
f  = h*v;      % f = front wheel rolling distance
               % b = back wheel rolling distance
b  = d + f.*cos(w) - sqrt(d^2 - (f.*sin(w)).^2);
               % do = change in car angle
do = asin(sin(w).*f/d);

dy = [tt(b, z); do; h*a];   % change in state
y  = x + dy;                % new state

d5 = 1-power(y(1,:)+1,2)-power(y(2,:)+3,2); % obstacle 1 constraint 
d6 = 1-power(y(1,:),2)-power(y(2,:)-3,2); % obstacle 2 constraint 
d=[d5];


function y = sabs(x,p)
% smooth absolute-value function (a.k.a pseudo-Huber)
y = pp( sqrt(pp(x.^2,p.^2)), -p);


function [f,c,d,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu, gu, gx] = car_dyn_cst(x,u,full_DDP)
% combine car dynamics and cost
% use helper function finite_difference() to compute derivatives

if nargout == 2 %%% nargout -> 이 함수의 출력 단자가 2개인 경우
    f = car_dynamics(x,u);
    c = car_cost(x,u);
elseif nargout == 3
    ix = 1:4;
    iu = 5:6;
    
    d = car_constraint(x,u);
    xu_const = @(xu) car_constraint(xu(ix,:),xu(iu,:));
    J = finite_difference(xu_const, [x; u]);
    J       = squeeze(J);
    gx      = J(:,ix,:);
    gu      = J(:,iu,:);
  
    % some bug. 
    f = d;
    c = gu;
    d = gx;
    
else
    % state and control indices
    ix = 1:4;
    iu = 5:6;
    
    
    % dynamics first derivatives
    % f(x,u)를 x와 u로 미분하자. 수식 (2)~(4) 참조.
    xu_dyn  = @(xu) car_dynamics(xu(ix,:),xu(iu,:));
    J       = finite_difference(xu_dyn, [x; u]);
    fx      = J(:,ix,:); % 4 by 4 by 501 
    fu      = J(:,iu,:); % 4 by 2 by 501
     
    % dynamics second derivatives
    if full_DDP
        xu_Jcst = @(xu) finite_difference(xu_dyn, xu);
        JJ      = finite_difference(xu_Jcst, [x; u]);
        JJ      = reshape(JJ, [4 6 size(J)]);
        JJ      = 0.5*(JJ + permute(JJ,[1 3 2 4])); %symmetrize
        fxx     = JJ(:,ix,ix,:);
        fxu     = JJ(:,ix,iu,:);
        fuu     = JJ(:,iu,iu,:);    
    else
        [fxx,fxu,fuu] = deal([]); %% fxx, fxu, fuu 를 deal로 처리함. 
    end    
    
    % cost first derivatives
    xu_cost = @(xu) car_cost(xu(ix,:),xu(iu,:));
    J = finite_difference(xu_cost, [x; u]);
    J       = squeeze(J);
    cx      = J(ix,:);
    cu      = J(iu,:);
    
    % cost second derivatives
    xu_Jcst = @(xu) squeeze(finite_difference(xu_cost, xu));
    JJ      = finite_difference(xu_Jcst, [x; u]);
    JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize
    cxx     = JJ(ix,ix,:);
    cxu     = JJ(ix,iu,:);
    cuu     = JJ(iu,iu,:);
    
    % constraint value and derivatives
    d = car_constraint(x,u);
    xu_const = @(xu) car_constraint(xu(ix,:),xu(iu,:));
    J = finite_difference(xu_const, [x; u]);
    J       = squeeze(J);
    gx      = J(:,ix,:);
    gu      = J(:,iu,:);
    
    [f,c] = deal([]);
end


function J = finite_difference(fun, x, h)
% simple finite-difference derivatives
% assumes the function fun() is vectorized
%%%% 식은 복잡하지만 h가 엄청 작을 때 f'(x) = f(x+h)-f(x) / h 임. 그거를 메트릭스에서도 사용가능하게 만든 버전임. 

if nargin < 3 %% 입력 단자가 3개보다 작은 경우, default h로 설정
    h = 2^-17;
end

[n, K]  = size(x); % x의 행과 열의 수 리턴. 아마도 n=6, K=1+T
H       = [zeros(n,1) h*eye(n)]; % H는 첫 열만 0, 나머지는 Identity Matrix, 아마도 6 by 7 matrix
H       = permute(H, [1 3 2]);  %H를 6 by 1 by 7로 나눈다. 
X       = pp(x, H); %H matrix의 각 행에 x를 더한다. 
X       = reshape(X, n, K*(n+1)); %6 by 1 by 7 matrix를 6 by (1+T)*6로 다시 치환함.
Y       = fun(X); %%각 행의 미소변화량으로 Y를 업데이트함. Y=4 by (1+T)*6
m       = numel(Y)/(K*(n+1)); %numel(Y) = array의 수 = 4*(1+T)*6, m이 4가 나와야 정상.
Y       = reshape(Y, m, K, n+1); % 4 by 1+T by 7
J       = pp(Y(:,:,2:end), -Y(:,:,1)) / h; % J=4 by 501 by 6
J       = permute(J, [1 3 2]);


% ======== graphics functions ========
function h = car_plot(x,u)

body        = [0.9 2.1 0.3];           % body = [width length curvature]
bodycolor   = 0.5*[1 1 1];
headlights  = [0.25 0.1 .1 body(1)/2]; % headlights [width length curvature x]
lightcolor  = [1 1 0];
wheel       = [0.15 0.4 .06 1.1*body(1) -1.1 .9];  % wheels = [width length curvature x yb yf]
wheelcolor  = 'k';

h = [];

% make wheels
for front = 1:2
   for right = [-1 1]
      h(end+1) = rrect(wheel,wheelcolor)'; %#ok<AGROW>
      if front == 2
         twist(h(end),0,0,u(1))
      end
      twist(h(end),right*wheel(4),wheel(4+front))
   end
end

% make body
h(end+1) = rrect(body,bodycolor);

% make window (hard coded)
h(end+1) = patch([-.8 .8 .7 -.7],.6+.3*[1 1 -1 -1],'w');

% headlights
h(end+1) = rrect(headlights(1:3),lightcolor);
twist(h(end),headlights(4),body(2)-headlights(2))
h(end+1) = rrect(headlights(1:3),lightcolor);
twist(h(end),-headlights(4),body(2)-headlights(2))

% put rear wheels at (0,0)
twist(h,0,-wheel(5))
rectangle('Position', [-2 -4 2 2], 'Curvature', [1 1], 'FaceColor', [0 .5 .5])
rectangle('Position', [-1 2 2 2], 'Curvature', [1 1], 'FaceColor', [0 .5 .5])
% align to x-axis
twist(h,0,0,-pi/2)

% make origin (hard coded)
ol = 0.1;
ow = 0.01;
h(end+1) = patch(ol*[-1 1 1 -1],ow*[1 1 -1 -1],'k');
h(end+1) = patch(ow*[1 1 -1 -1],ol*[-1 1 1 -1],'k');

twist(h,x(1),x(2),x(3))

function twist(obj,x,y,theta)
% a planar twist: rotate object by theta, then translate by (x,y)
i = 1i;
if nargin == 3
   theta = 0;
end
for h = obj
   Z = get(h,'xdata') + i*get(h,'ydata');
   Z = Z * exp(i*theta);
   Z = Z + (x + i*y);
   set(h,'xdata',real(Z),'ydata',imag(Z));
end

function h = rrect(wlc, color)
% draw a rounded rectangle (using complex numbers and a kronecker sum :-)

N        = 25; % number of points per corner

width    = wlc(1);
length   = wlc(2);
curve    = wlc(3);

a        = linspace(0,2*pi,4*N);
circle   = curve*exp(1i*a);
width    = width-curve;
length   = length-curve;
rect1    = diag(width*[1 -1 -1 1] + 1i*length *[1 1 -1 -1]);
rectN    = sum(kron(rect1, ones(1,N)), 1) ;
rr       = circle + rectN;
rr       = [rr rr(1)]; % close the curve

h        = patch(real(rr),imag(rr),color);

% utility functions: singleton-expanded addition and multiplication
function c = pp(a,b)
c = bsxfun(@plus,a,b);

function c = tt(a,b)
c = bsxfun(@times,a,b);