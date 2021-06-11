clear;
close;
clc;

%% Reading the dataset
data = importdata('heart.dat', ' ');
samples = data(:,1:end-1);
target = data(:,end);

%% Initializaing Variables
figure(1);
[instances,attributes] = size(samples);
for s = 10:10:instances
    disp(['Number of training samples:', num2str(s)])
    gmms = [];
    belongs_to_gmm = [];

    kernel_count = 0;
    receptive_ini = 0.8;
    t1 = 0.5;
    t2 = 0.0033;
    nmax=100;
    ro = 100;
    o=0;
    
    %% The Main loop
    for i=1:s
        % current_sample stores the sample going throught the network at
        % current time. current_target represents the target class of the sample
        current_sample = samples(i,:);
        current_target = target(i);
        n = i;
        
        % if the sample is not from an existent class in the neural network
        target_gmm_index = find(gmms == current_target);
        if isempty(target_gmm_index)
            kernel_count = kernel_count + 1;
            belongs_to_gmm(kernel_count, 1) = current_target;
            
            % if this is the first kernel of the first GMM
            if isempty(gmms)
                % setting parameters of first kernel using 22
                w(kernel_count,1) = 1;
                receptive(kernel_count,1) = receptive_ini;
                for j=1:attributes
                    mu(kernel_count,j) = current_sample(j);
                    sigma2(kernel_count,j) = 1;
                end
                % if this is the first kernel of a GMM but other gmms already exist
            else
                w(kernel_count,1) = 1;
                % setting receptive field size using 19
                numerator = 0;
                denominator = 0;
                for k=1:kernel_count-1
                    numerator = numerator + w(k,1)*receptive(k,1);
                    denominator = denominator + w(k,1);
                end
                receptive(kernel_count,1) = numerator/denominator;
                % setting Variance and mean vectors using 17 and 18
                numerator = 0;
                denominator = 0;
                for j=1:attributes
                    % using 17
                    mu(kernel_count,j) = current_sample(j);
                    % using 18
                    for k=1:kernel_count-1
                        numerator = numerator + w(k,1)*sigma2(k,j);
                        denominator = denominator + w(k,1);
                    end
                    sigma2(kernel_count,j) = numerator/denominator;
                end
            end
            gmms(end+1) = current_target;
            
            % If the sample is from a class that is already in the neural network
        else
            gmm_count = size(gmms,2);
            target_class_kernels = find(belongs_to_gmm==current_target);
            for j=1:kernel_count
                A = inv(diag(sqrt(sigma2(j,:))));
                xp = A*current_sample';
                xp = xp/norm(xp);
                mup = A*mu(j,:)';
                mup = mup/norm(mup);
                dist(j,1) = xp'*mup - 1;
                gaussian(j,1) = (1/(sqrt(2*pi*receptive(j,1))))*exp(dist(j,1)/receptive(j,1));
            end
            
            for j=1:gmm_count
                gmm_kernels = find(belongs_to_gmm==gmms(j));
                %gmm_kernels = pattern{gmms(j),1};
                gmm_output(j,1) = w(gmm_kernels)'*gaussian(gmm_kernels);
                if gmms(j) == current_target
                    target_class_output = gmm_output(j,1);
                end
            end
            
            for j=1:kernel_count
                gmm_index = find(gmms==belongs_to_gmm(j,1));
                posterior(j,1) = (w(j,1)*gaussian(j,1))/gmm_output(gmm_index,1);
            end
            
            gmm_output = gmm_output/norm(gmm_output,1);
            target_class_output = target_class_output/norm(gmm_output,1);
            
            % Check two kernels in the same class with two largest posteriors
            if size(target_class_kernels,1) >= 2
                [post_sorted,post_sorted_index] = sort(posterior(target_class_kernels));
                post_sorted_index = target_class_kernels(post_sorted_index);
                max_kernel = post_sorted(end,1);
                max_kernel_index = post_sorted_index(end,1);
                second_max_kernel = post_sorted(end-1,1);
                second_max_kernel_index = post_sorted_index(end-1,1);
                
                f_success = 1;
                %f-test
                for j=1:attributes
                    f_test = sigma2(max_kernel_index,j)/sigma2(second_max_kernel_index,j);
                    if (f_test >= 1.945)
                        f_success = 0;
                        break
                    end
                end
                
                t_success = 1;
                if f_success == 1
                    for j=1:attributes
                        common_sigma = sqrt(sigma2(max_kernel_index,1))*sqrt(sigma2(second_max_kernel_index,1));
                        t_test = (sqrt(n*w(max_kernel_index,1)))*((mu(max_kernel_index,1)-mu(second_max_kernel_index,1))/(sqrt(2)*sqrt(common_sigma)));
                        if (t_test >= 3.291) || (t_test <= -3.291)
                            t_success = 0;
                            break
                        end
                    end
                end
                
                if (f_success == 1) && (t_success == 1)
                    wr = w(max_kernel_index,1) + w(second_max_kernel_index,1);
                    mus = mean(mu(max_kernel_index,:));
                    mut = mean(mu(second_max_kernel_index,:));
                    for j=1:attributes
                        mur(1,j) = ((w(max_kernel_index,1)*mu(max_kernel_index,j))+((w(second_max_kernel_index,1)*mu(second_max_kernel_index,j))))/wr;
                        sigmar(1,j) = (((w(max_kernel_index,1)*(sigma2(max_kernel_index,j)-(mus^2)))+(w(second_max_kernel_index,1)*(sigma2(second_max_kernel_index,j)-(mup^2))))/wr) - (mur^2);
                        recr = ((w(max_kernel_index,1)*receptive(max_kernel_index,1))+(w(second_max_kernel_index,1)*receptive(second_max_kernel_index,1)))/wr;
                    end
                    w(max_kernel_index,1) = [];
                    receptive(max_kernel_index,1) = [];
                    mu(max_kernel_indexj,:) = [];
                    sigma2(max_kernel_index,:) = [];
                    belongs_to_gmm(max_kernel_index,1) = [];
                    
                    w(second_max_kernel_index,1) = [];
                    receptive(second_max_kernel_index,1) = [];
                    mu(second_max_kernel_index,:) = [];
                    sigma2(second_max_kernel_index,:) = [];
                    belongs_to_gmm(second_max_kernel_index,1) = [];
                    
                    kernel_count = kernel_count - 1;
                    
                    w(kernel_count,1) = wr;
                    receptive(kernel_count,1) = recr;
                    mu(kernel_count,:) = mur(1,:);
                    sigma2(kernel_count,:) = sigmar(1,:);
                    belongs_to_gmm(kernel_count,1) = current_target;
                    
                    
                end
            end
            
            
            if target_class_output >= t1
                for j=1:size(target_class_kernels,1)
                    index = target_class_kernels(j,1);
                    for k=1:attributes
                        mu(index,k) = mu(index,k) + ((posterior(index,1)*(current_sample(k)-mu(index,k)))/n*w(index,1));
                        sigma2(index,k) = sigma2(index,k) + ((posterior(index,1)*(((current_sample(k)-mu(index,k))^2)-sigma2(index,k)))/n*w(index,1));
                        w(index,1) = w(index,1) + ((posterior(index,1)-w(index,1))/n);
                    end
                end
            end
            
            % if the largest output is not from the same class
            [gmm_sorted,gmm_sorted_index] = sort(gmm_output);
            gmm_sorted_index = gmms(gmm_sorted_index);
            largest_gmm_index = gmm_sorted_index(end);
            
            if largest_gmm_index ~= current_target
                o = o + 1;
                largest_kernels = find(belongs_to_gmm==largest_gmm_index);
                largest_gmm_dist = dist(largest_kernels);
                target_gmm_dist = dist(target_class_kernels);
                [q,v] = min(largest_gmm_dist);
                [r,e] = min(target_gmm_dist);
                v = largest_kernels(v);
                e = target_class_kernels(e);
                dist_e = -2*dist(v,1);
                prev_rec = receptive(v,1);
                receptive(v,1) = receptive(v,1) + ((posterior(v,1)*(dist_e - receptive(v,1)))/(ro*w(v,1)));
                if prev_rec < receptive(v,1)
                    signal = -1;
                else
                    signal = 1;
                end
                receptive(e,1) = receptive(e,1) + ((posterior(e,1)*(dist_e - receptive(e,1)))/(signal*ro*w(v,1)));
                
                
            end
            
            % when output is less than threshold t1, add a new kernel to the
            % same class as sample and set the parameters
            if target_class_output < t1
                kernel_count = kernel_count + 1;
                belongs_to_gmm(kernel_count, 1) = current_target;
                
                % setting receptive field size using 19
                numerator = 0;
                denominator = 0;
                for k=1:size(target_class_kernels,1)
                    kernel_num = target_class_kernels(k);
                    numerator = numerator + w(kernel_num,1)*receptive(kernel_num,1);
                    denominator = denominator + w(kernel_num,1);
                end
                receptive(kernel_count,1) = numerator/denominator;
                
                numerator = 0;
                denominator = 0;
                for j=1:attributes
                    mu(kernel_count,j) = current_sample(j);
                    for k=1:size(target_class_kernels,1)
                        kernel_num = target_class_kernels(k);
                        numerator = numerator + w(kernel_num,1)*sigma2(kernel_num,j);
                        denominator = denominator + w(kernel_num,1);
                    end
                    sigma2(kernel_count,j) = numerator/denominator;
                end
                w(kernel_count,1) = 1/n;
                for j=1:size(target_class_kernels,1)
                    kernel_num = target_class_kernels(j);
                    w(kernel_num,1) = w(kernel_num,1) - (1/(n*size(target_class_kernels,1)));
                end
            end
            
            % remove all weights making little contribution to network
            should_delete = 0;
            for j=1:kernel_count
                if w(j,1) < t2
                    should_delete(end,1) = j;
                end
            end
            for j=1:size(should_delete,1)
                del = should_delete(j,1);
                if del == 0
                    break
                end
                w(del,:) = [];
                receptive(del,:) = [];
                mu(del,:) = [];
                sigma2(del,:) = [];
                belongs_to_gmm(del,:) = [];
                kernel_count = kernel_count - 1;
            end
        end
    end
    disp(['error:', num2str(mean(posterior))])
    figure(1);
    plot(s,mean(posterior),'Ob')
    title('Heart Data Base')
    xlabel('Number of training samples')
    ylabel('error')
    hold on
    figure(2);
    plot(s,numel(w),'*r')
    title('Heart Data Base')
    xlabel('Number of training samples')
    ylabel('Number of weights')
    hold on
end
