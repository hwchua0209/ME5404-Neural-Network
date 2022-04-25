clc;
clear;

figure();
x  = linspace(0,100,1000);
y1 = 1 ./ x;
y2 = 100 ./ (100 + x);
y3 = (1 + log(x)) ./ x;
y4 = (1 + 5 .* log(x)) ./ x;
y5 = 0.005;

plot(x, y1, 'LineWidth', 1.5);
grid on;
hold on;
plot(x, y2, 'LineWidth', 1.5);
plot(x, y3, 'LineWidth', 1.5);
plot(x, y4, 'LineWidth', 1.5);
xlim([0, 100]);
ylim([0, 2]);
line(xlim,[0.005, 0.005], 'LineWidth', 1.5, 'color', 'k');
legend('1/x', '100/(100+x)', '(1+log(x))/x', '(1+5log(x))/x', 'Cut-off 0.005');
xlabel('Timestep, k');
ylabel('Epsilon');
title('Plot of Various Epilon Greedy Decay Method');
hold off;