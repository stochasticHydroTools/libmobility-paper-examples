clear;
% close all;
grav_height = 61

if grav_height == 61
  grav_fname = "hg61/";
elseif grav_height == 15
  grav_fname = "hg15/";
end

figure()

set(groot, 'defaultAxesLabelFontSizeMultiplier', 1.2);
set(groot, 'defaultAxesFontSize', 40);  % Base font size

a=0.656;

colors = ["#6495ed",
"#ff77ff",
"#002395",
"#a20983"];

lw = 4;
lw_ref=6;
ms = 1;
m_freq = 15;

i=1;
pos_leg_handles = [];
height_leg_handles = [];
for state = ["deterministic/", "stochastic/"]
  dir = "../" + grav_fname + state + "output/";
  ref_dir = "../extracted_data/" + grav_fname + state;

  c = colors(i);
  ref_c = colors(i + 2);

  subplot(1, 2, 2); % DIST PLOT
  hold on
  p_dist = readmatrix(dir + "pos_dist.txt");
  p_start = readmatrix(dir + "start_pos_dist.txt");
  p_dist_ref = readmatrix(ref_dir + "pos_dist.csv");

  t_star = readmatrix(dir + "t_star.txt");
  mean(t_star);

  if state == "deterministic/"
    dist_start_handle = plot(p_start(:,1), p_start(:,2), color='black', linestyle='-', linewidth=3);
    pos_leg_handles(5) = dist_start_handle;
  end
  dist_handle = plot(p_dist(:,1), p_dist(:,2), color=c, linewidth=lw);
  dist_ref_handle = plot(p_dist_ref(:,1), p_dist_ref(:,2),color=ref_c, LineStyle=":", LineWidth=lw_ref);

  pos_leg_handles(i) = dist_handle;
  pos_leg_handles(i+2) = dist_ref_handle;
  xlim([0,max(p_dist(:,1))])

  h_dist = readmatrix(dir + "heights_dist.txt");
  h_start = readmatrix(dir + "start_heights_dist.txt");
  h_dist_ref = readmatrix(ref_dir + "heights_dist.csv");

  subplot(1, 2, 1); % HEIGHT PLOT
  hold on
  if state == "stochastic/"
    heights_start_handle = plot(h_start(:,1), h_start(:,2), color='black', linestyle='-', linewidth=3);
    height_leg_handles(5) = heights_start_handle;
  end
  height_handle = plot(h_dist(:,1), h_dist(:,2), color=c, linewidth=lw);
  height_ref_handle = plot(h_dist_ref(:,1), h_dist_ref(:,2), color=ref_c, LineStyle=":", linewidth=lw_ref);

  height_leg_handles(i) = height_handle;
  height_leg_handles(i+2) = height_ref_handle;
  xlim([0, max(h_dist(:,1))])
  ylim([0,inf])

  i = i + 1;
end

subplot(1, 2, 1); % HEIGHTS LABELS
xlabel("$h/a$")
ylabel("$P(h) \times a$")
legend(height_leg_handles, ["Deterministic", "Stochastic", "Deterministic (ref.)", "Stochastic (ref.)", "Initial configuration"]);
box on

subplot(1,2,2) % POSITIONS LABELS
xlabel("$x/a$")
ylabel("$\rho(x) \times a$")
box on








%%%%%%%%%%%%%%%%%%%%%%%% AXIS
subplot(1,2,1) % HEIGHTS AXIS STUFF
ax = gca;
ax.XTickLabelRotation = 0;
ax.YTickLabelRotation = 0;

if grav_height == 15
yt = 0:0.1:1.2;
labels = repmat({''}, size(yt));  % start with all blank labels
labels(ismember(yt, [0, 0.5, 1])) = {'0', '0.5', '1'};  % assign only selected labels
yticks(yt)
yticklabels(labels)

xt = 0:2:25;
labels = repmat({''}, size(xt));
labels(ismember(xt, [0 10 20])) = {'0', '10', '20'};
xticks(xt)
xticklabels(labels)

daspect([25,1.2,1])
else % hg61
yt = 0:0.01:0.16;
labels = repmat({''}, size(yt));
labels(ismember(yt, [0, 0.05, 0.1, 0.15])) = {'0', '0.05', '0.1', '0.15'};
yticks(yt)
yticklabels(labels)

xt = 0:5:50;
labels = repmat({''}, size(xt));
labels(ismember(xt, [0 20 40])) = {'0', '20', '40'};
xticks(xt)
xticklabels(labels)

daspect([50,0.16,1])
ylim([0, 0.16])
end

subplot(1,2,2); % POSITIONS AXIS STUFF
ax = gca;
ax.XTickLabelRotation = 0;
ax.YTickLabelRotation = 0;

yt = 0:0.02:0.2;
yticks(yt);
yt_labels = repmat({''}, size(yt));
yt_labels(ismember(yt, [0, 0.1, 0.2])) = {'0', '0.1', '0.2'};
yticklabels(yt_labels);

if grav_height == 15
    xt = 0:50:400;
    xt_labels = repmat({''}, size(xt));
    xt_labels(ismember(xt, [0, 200, 400])) = {'0', '200', '400'};

    daspect([400,0.2,1])
else
    xt = 0:50:500;
    xt_labels = repmat({''}, size(xt));
    xt_labels(ismember(xt, [0, 200, 400])) = {'0', '200', '400'};

    daspect([500,0.2,1])
end

xticks(xt);
xticklabels(xt_labels);