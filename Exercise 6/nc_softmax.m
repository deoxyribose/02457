function probs = nc_softmax(phi)

%NC_SOFTMAX    Neural classifier softmax
%  probs = nc_softmax(phi)
%  Carry out the softmax operation
%  Input:  
%      phi: Matrix of outputs of the network from NC_FORWARD
%           rows are the individual output neurons.
%  Output:  
%    probs: Matrix of posterior probabilities. Each row is the
%           are individual class prob for a specific example.
% 
%  Neural classifier, DSP IMM DTU, JL97

%  cvs: $Revision: 1.2 $

% Determine the number of examples
exam=size(phi,1);

% Expand output unit outputs with 'dummy' zeros for the last class
 phi = [phi zeros(exam,1)];

c=size(phi,2);

% Offsets for numerical trick
offsets = max(phi')';

% Apply exp() to output unit outputs (using numerical trick)
phi = exp(phi-kron(ones(1,c),offsets));

  
% Denominators for the softmax  
tmp = sum(phi')';
denom = ones(size(tmp)) ./ tmp;


% Compute softmax probabilities for each example
probs=phi.*kron(ones(1,c),denom);




