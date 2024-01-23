function [running] = check_running(self)
    
    running = exist('MainWindow', 'file') == 1;

end