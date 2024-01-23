function python_get_ebfret_instance()

[status, result] = system('pgrep ebFRET()');

if status == 0
    disp('ebFRET is running.');
else
    disp('ebFRET is not running.');
end