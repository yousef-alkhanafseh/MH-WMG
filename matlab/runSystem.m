%% Updated Code Without Parallel Processing (Serial Execution) – Fixed for full 5-second simulation

% Define the Simulink model name and simulation stop time
modelName = 'KundurTwoAreaSystemModified';
maxSimTime = 5;  % seconds

% Define lengths and devices for each area (A1, A2, A3, A4, A5, A6)
lengths = struct(...
    'A1', [0, 25], ...
    'A2', [0, 10], ...
    'A3', [0, 25], ...
    'A4', [0, 10], ...
    'A5', [0, 110], ...
    'A6', [0, 110] ...
);


devices = struct(...
    'A1', {{'KundurTwoAreaSystem_BACKUP_NEW/Area 2/25km Area 2a', ...
             'KundurTwoAreaSystem_BACKUP_NEW/Area 2/25km Area 2b'}}, ...
    'A2', {{'KundurTwoAreaSystem_BACKUP_NEW/Area 2/10km Area 2a', ...
             'KundurTwoAreaSystem_BACKUP_NEW/Area 2/10km Area 2b'}}, ...
    'A3', {{'KundurTwoAreaSystem_BACKUP_NEW/Area 1/25km Area 1a', ...
            'KundurTwoAreaSystem_BACKUP_NEW/Area 1/25km Area 1b'}}, ...
    'A4', {{'KundurTwoAreaSystem_BACKUP_NEW/Area 1/10km Area 1a', ...
             'KundurTwoAreaSystem_BACKUP_NEW/Area 1/10km Area 1b'}}, ...
    'A5', {{'KundurTwoAreaSystem_BACKUP_NEW/DPL 71a', ...
            'KundurTwoAreaSystem_BACKUP_NEW/DPL 71b'}}, ...
    'A6', {{'KundurTwoAreaSystem_BACKUP_NEW/DPL 81a', ...
            'KundurTwoAreaSystem_BACKUP_NEW/DPL 81b'}} ...
);

% Define fault combinations (each row: FaultA, FaultB, FaultC, GroundFault)
combinations = {...
    'on', 'on', 'on', 'on';
    'on', 'on', 'on', 'off'; 
    'on', 'on', 'off', 'on'; 
    'on', 'on', 'off', 'off'; 
    'on', 'off', 'on', 'on'; 
    'on', 'off', 'on', 'off'; 
    'on', 'off', 'off', 'on'; 
    'off', 'on', 'on', 'on'; 
    'off', 'on', 'on', 'off'; 
    'off', 'on', 'off', 'on'; 
    'off', 'off', 'on', 'on';
    'off', 'off', 'off', 'off'};

areaNames = fieldnames(lengths);
for d = 1:numel(areaNames)
    area = areaNames{d};
    lengths_a = lengths.(area)(1);
    lengths_b = lengths.(area)(2);

    % Determine incremental step based on area type
    if ismember(area, {'A1', 'A3'})
        incremental_length = 0.5;
    elseif ismember(area, {'A2', 'A4'})
        incremental_length = 0.2;
    elseif ismember(area, {'A5', 'A6'})
        incremental_length = 2.2;
    end

    while lengths_b > incremental_length
        % Set Goto Tag names based on the area
        if strcmp(area, 'A1')
            v_m_goto = 'Vabc_25_2a';
            i_m_goto = 'Iabc_25_2a';
        elseif strcmp(area, 'A2')
            v_m_goto = 'Vabc_10_2a';
            i_m_goto = 'Iabc_10_2a';
        elseif strcmp(area, 'A3')
            v_m_goto = 'Vabc_25_1a';
            i_m_goto = 'Iabc_25_1a';
        elseif strcmp(area, 'A4')
            v_m_goto = 'Vabc_10_1a';
            i_m_goto = 'Iabc_10_1a';
        elseif strcmp(area, 'A5')
            v_m_goto = 'Vabc_7a';
            i_m_goto = 'Iabc_7a';
        elseif strcmp(area, 'A6')
            v_m_goto = 'Vabc_8a';
            i_m_goto = 'Iabc_8a';
        end

        % Set fault block names, sources, and scopes based on the area
        if ismember(area, {'A1', 'A2'})
            three_ph_fault = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2/Three-Phase Fault';
            area_main_name = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2';
            faulted_vi_scope = 'ScopeDataFaultedVI_2';
            faulted_v_scope  = 'ScopeDataFaultedV_2';
            faulted_i_scope  = 'ScopeDataFaultedI_2';

            vi_v_source = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2/From7';
            vi_i_source = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2/From8';
            v_source    = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2/From1';
            i_source    = 'KundurTwoAreaSystem_BACKUP_NEW/Area 2/From2';
        elseif ismember(area, {'A3', 'A4'})
            three_ph_fault = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1/Three-Phase Fault1';
            area_main_name = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1';
            faulted_vi_scope = 'ScopeDataFaultedVI_1';
            faulted_v_scope  = 'ScopeDataFaultedV_1';
            faulted_i_scope  = 'ScopeDataFaultedI_1';

            vi_v_source = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1/From7';
            vi_i_source = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1/From8';
            v_source    = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1/From1';
            i_source    = 'KundurTwoAreaSystem_BACKUP_NEW/Area 1/From3';
        elseif ismember(area, {'A5', 'A6'})
            three_ph_fault = 'KundurTwoAreaSystem_BACKUP_NEW/Three-Phase Fault';
            area_main_name = 'KundurTwoAreaSystem_BACKUP_NEW';
            faulted_vi_scope = 'ScopeDataFaultedVI_0';
            faulted_v_scope  = 'ScopeDataFaultedV_0';
            faulted_i_scope  = 'ScopeDataFaultedI_0';

            vi_v_source = 'KundurTwoAreaSystem_BACKUP_NEW/From11';
            vi_i_source = 'KundurTwoAreaSystem_BACKUP_NEW/From12';
            v_source    = 'KundurTwoAreaSystem_BACKUP_NEW/From7';
            i_source    = 'KundurTwoAreaSystem_BACKUP_NEW/From8';
        end

        % Update the length values for this iteration
        lengths_a = lengths_a + incremental_length;
        lengths_b = lengths_b - incremental_length;

        % Number of fault combinations to simulate
        numComb = size(combinations, 1);

        % Use a standard for loop to run fault combinations serially
        parfor i = 1:numComb
            % Reload the model to ensure a clean state
            load_system(modelName);
            set_param(modelName, 'StopTime', num2str(maxSimTime));
            cs = getActiveConfigSet(modelName);
            set_param(cs, 'StopTime', '5'); 

            % Reapply device lengths and Goto Tags
            set_param(devices.(area){1}, 'Length', num2str(lengths_a));
            set_param(devices.(area){2}, 'Length', num2str(lengths_b));

            set_param(vi_v_source, 'GotoTag', v_m_goto);
            set_param(vi_i_source, 'GotoTag', i_m_goto);
            set_param(v_source,    'GotoTag', v_m_goto);
            set_param(i_source,    'GotoTag', i_m_goto);

            % Get port handles for connecting blocks
            dpl_param = get_param(devices.(area){1}, 'PortHandles');
            ph3_param = get_param(three_ph_fault, 'PortHandles');

            % Add connection lines
            add_line(area_main_name, dpl_param.RConn(1), ph3_param.LConn(1), 'autorouting', 'on');
            add_line(area_main_name, dpl_param.RConn(2), ph3_param.LConn(2), 'autorouting', 'on');
            add_line(area_main_name, dpl_param.RConn(3), ph3_param.LConn(3), 'autorouting', 'on');
            
            % Force model update to process changes
%             set_param(modelName, 'SimulationCommand', 'update');

            % Set fault parameters for this iteration
            fault_a = combinations{i, 1};
            fault_b = combinations{i, 2};
            fault_c = combinations{i, 3};
            fault_ground = combinations{i, 4};

            set_param(three_ph_fault, 'FaultA', fault_a);
            set_param(three_ph_fault, 'FaultB', fault_b);
            set_param(three_ph_fault, 'FaultC', fault_c);
            set_param(three_ph_fault, 'GroundFault', fault_ground);
            set_param(three_ph_fault, 'FaultResistance', '0.001');
            set_param(three_ph_fault, 'SnubberResistance', '1e6');
            set_param(three_ph_fault, 'GroundResistance', '0.01');
            set_param(three_ph_fault, 'SwitchTimes', '[2, 2.1]');

            % Run the simulation and retrieve outputs
            simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');

            % Access and save voltage data from scope (pmuScopeData1)
            vabc_g1_data = simOut.get('pmuScopeData1');
            vabc_g1_time = vabc_g1_data.time;
            vabc_g1_magnitude = vabc_g1_data.signals(1).values;
            vabc_g1_angle = vabc_g1_data.signals(2).values;
            vabc_g1_f = vabc_g1_data.signals(3).values;

            vabc_g1_filename = sprintf('data/timeseries/%s/voltage/%s_%s_%s_%s_%s.csv', ...
                                        area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            vabc_g1_table = table(vabc_g1_time, vabc_g1_magnitude, vabc_g1_angle, vabc_g1_f, ...
                                   'VariableNames', {'time','vabc_g1_magnitude', 'vabc_g1_angle', 'vabc_g1_f'});
            writetable(vabc_g1_table, vabc_g1_filename);

            % Access current scope data (pmuScopeData2)
            iabc_g1_data = simOut.get('pmuScopeData2');
            iabc_g1_time = iabc_g1_data.time;
            iabc_g1_magnitude = iabc_g1_data.signals(1).values;
            iabc_g1_angle = iabc_g1_data.signals(2).values;
            iabc_g1_f = iabc_g1_data.signals(3).values;

            iabc_g1_filename = sprintf('data/timeseries/%s/current/%s_%s_%s_%s_%s.csv', ...
                                        area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            iabc_g1_table = table(iabc_g1_time, iabc_g1_magnitude, iabc_g1_angle, iabc_g1_f, ...
                                   'VariableNames', {'time','iabc_g1_magnitude', 'iabc_g1_angle', 'iabc_g1_f'});
            writetable(iabc_g1_table, iabc_g1_filename);

            % Access power scope data (ScopeData1)
            p_g1_data = simOut.get('ScopeData1');
            p_g1_time = p_g1_data.time;
            p_g1_magnitude = p_g1_data.signals(1).values;
            p_g1_filename = sprintf('data/timeseries/%s/power/%s_%s_%s_%s_%s.csv', ...
                                    area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            p_g1_table = table(p_g1_time, p_g1_magnitude, 'VariableNames', {'time','p_g1_magnitude'});
            writetable(p_g1_table, p_g1_filename);

            % Access b1 data (mach1)
            b1_data = simOut.get('mach1');
            b1_time = b1_data.time;
            b1_v = b1_data.signals(1).values;
            b1_i = b1_data.signals(2).values;
            b1_filename = sprintf('data/timeseries/%s/b1/%s_%s_%s_%s_%s.csv', ...
                                  area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            b1_table = table(b1_time, b1_v, b1_i, 'VariableNames', {'time', 'b1_v', 'b1_i'});
            writetable(b1_table, b1_filename);

            % Access faulted VI scope data
            faulted_vi_data = simOut.get(faulted_vi_scope);
            test_time = faulted_vi_data.time;
            faulted_vi_v = faulted_vi_data.signals(1).values;
            faulted_vi_i = faulted_vi_data.signals(2).values;
            faulted_vi_filename = sprintf('data/timeseries/%s/faulted_vi/%s_%s_%s_%s_%s.csv', ...
                                          area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            faulted_vi_table = table(test_time, faulted_vi_v, faulted_vi_i, ...
                                     'VariableNames', {'time', 'faulted_v', 'faulted_i'});
            writetable(faulted_vi_table, faulted_vi_filename);

            % Access faulted V scope data
            test_v_data = simOut.get(faulted_v_scope);
            test_v_time = test_v_data.time;
            test_v_m = test_v_data.signals(1).values;
            test_v_a = test_v_data.signals(2).values;
            test_v_f = test_v_data.signals(3).values;
            test_v_filename = sprintf('data/timeseries/%s/faulted_pmu_v/%s_%s_%s_%s_%s.csv', ...
                                      area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            test_v_table = table(test_v_time, test_v_m, test_v_a, test_v_f, ...
                                 'VariableNames', {'time', 'faulted_pmu_v_m', 'faulted_pmu_v_a', 'faulted_pmu_v_f'});
            writetable(test_v_table, test_v_filename);

            % Access faulted I scope data
            test_i_data = simOut.get(faulted_i_scope);
            test_i_time = test_i_data.time;
            test_i_m = test_i_data.signals(1).values;
            test_i_a = test_i_data.signals(2).values;
            test_i_f = test_i_data.signals(3).values;
            test_i_filename = sprintf('data/timeseries/%s/faulted_pmu_i/%s_%s_%s_%s_%s.csv', ...
                                      area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            test_i_table = table(test_i_time, test_i_m, test_i_a, test_i_f, ...
                                 'VariableNames', {'time', 'faulted_pmu_i_m', 'faulted_pmu_i_a', 'faulted_pmu_i_f'});
            writetable(test_i_table, test_i_filename);

            % Access b2 data (mach2)
            b2_data = simOut.get('mach2');
            b2_time = b2_data.time;
            b2_v = b2_data.signals(1).values;
            b2_i = b2_data.signals(2).values;
            b2_filename = sprintf('data/timeseries/%s/b2/%s_%s_%s_%s_%s.csv', ...
                                  area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            b2_table = table(b2_time, b2_v, b2_i, 'VariableNames', {'time', 'b2_v', 'b2_i'});
            writetable(b2_table, b2_filename);

            % Access machines data (mach3)
            machines_data = simOut.get('mach3');
            machines_time = machines_data.time;
            machines_dtheta = machines_data.signals(1).values;
            machines_w = machines_data.signals(2).values;
            machines_pa = machines_data.signals(3).values;
            machines_vt = machines_data.signals(4).values;
            machines_filename = sprintf('data/timeseries/%s/machines/%s_%s_%s_%s_%s.csv', ...
                                        area, num2str(lengths_a), fault_a, fault_b, fault_c, fault_ground);
            machines_table = table(machines_time, machines_dtheta, machines_w, machines_pa, machines_vt, ...
                                   'VariableNames', {'time', 'machines_dtheta', 'machines_w', 'machines_pa', 'machines_vt'});
            writetable(machines_table, machines_filename);

            % Close the model without saving changes
            close_system(modelName, 0);
        end
    end
end
