function close(self)

    if exist('self.handles.mainWindow', 'var')
    % close ui elements
    % (we are assuming matlab deletes dependent ui elements correctly)
        delete(self.handles.mainWindow);
    end
    % delete containing class structure
    delete(self);
end
