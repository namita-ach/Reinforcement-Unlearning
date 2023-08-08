# Reinforcement-Unlearning
Machine unlearning refers to the process of mitigating the influence of specific training data on machine learning models based on removal requests from data owners. 
This practice has gained significant attention. 
However, one important area that has been largely overlooked in the research of unlearning is reinforcement learning. 
Reinforcement learning focuses on training an agent to make optimal decisions within an environment to maximize its cumulative rewards. During the training, the agent tends to memorize the features of the environment, which raises a significant concern of privacy. 
As per data protection regulations, the owner of the environment holds the right to revoke access to the agent's training data, thus necessitating the development of a novel and pressing research field,  known as reinforcement unlearning.
Reinforcement unlearning focuses on revoking entire environments rather than individual data samples. This unique characteristic presents three distinct challenges: 1) how to propose unlearning schemes for environments; 
2) how to avoid degrading the agent's performance in remaining environments; and 3) how to evaluate the effectiveness of unlearning. 
To tackle these challenges, we propose two reinforcement unlearning methods. The first method is based on decremental reinforcement learning, which aims to erase the agent's previously acquired knowledge gradually. 
The second method leverages environment poisoning attacks, which encourages the agent to learn new, albeit incorrect, knowledge to remove the unlearning environment. 
Particularly, to tackle the third challenge, we introduce the concept of ``environment reconstruction inference'' to evaluate the reinforcement unlearning outcomes. 
Through extensive experimentation, we demonstrate the effectiveness of our proposed methods in various scenarios. 


https://github.com/cp-lab-uts/Reinforcement-Unlearning/assets/64918395/6c898d3d-e274-4158-b6ce-1150bf2e2a5a

