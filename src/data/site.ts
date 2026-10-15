export const profile = {
  name: 'Amardeep Mehta',
  shortName: 'AM',
  role: 'AI Systems & Distributed Systems Researcher',
  headline: 'Building efficient distributed and accelerator-backed AI systems.',
  intro: 'I am a computer systems researcher and ML systems engineer with over 10 years of experience in distributed systems, cloud/edge platforms, LLM serving, GPU/NPU-backed runtimes and AI infrastructure. I currently lead ML infrastructure architecture at Lenovo AI Technology Center, following systems research at Huawei and Ericsson Research.',
  location: 'Edinburgh, United Kingdom',
  affiliation: 'Lenovo AI Technology Center',
  position: 'ML Infrastructure Lead',
  email: 'amar123deep@gmail.com',
  github: '', // Add your verified GitHub profile URL
  linkedin: 'https://www.linkedin.com/in/amardeep-mehta/',
  scholar: '', // Add your actual Google Scholar URL
  photo: '/images/amardeep.jpeg', // Add your portrait at public/images/profile.jpg
  cv: '/cv.pdf', // Public CV with phone number omitted
};
export const projects = [
  {slug:'multipath', number:'01', category:'GPU SYSTEMS / MODEL SERVING', title:'Multipath GPU Transfer', description:'Topology-aware parallel model-weight transfers using host-to-device and GPU peer-to-peer paths; CUDA/H200 transfer microbenchmarks and model-serving research.', tags:['CUDA','NVLink','LLM Serving','Performance']},
  {slug:'nefele', number:'02', category:'DISTRIBUTED SYSTEMS / ORCHESTRATION', title:'Nefele', description:'Decentralised process orchestration: cluster-wide process spawning, placement, naming, communication and lifecycle management.', tags:['Runtime Systems','Scheduling','Cloud / Edge']},
  {slug:'phd-research', number:'03', category:'DISTRIBUTED SYSTEMS / RESEARCH', title:'Resource Allocation for Mobile Edge Clouds', description:'PhD thesis on workload spikes, edge resource planning, cost-aware placement, scheduling and resource-constrained IoT applications.', tags:['Distributed Computing','Resource Allocation']},
];
export type Publication = {title:string; authors:string; venue:string; year:string; url?:string; type?:string};
export const publications: Publication[] = [
  {title:'Smart Edge Power Management to Improve Availability and Cost-efficiency of Edge Cloud', authors:'Amardeep Mehta and Lackis Eleftheriadis', venue:'IEEE CLOUD', year:'2022', url:'https://doi.org/10.1109/CLOUD55607.2022.00032'},
  {title:'Nefele: Process Orchestration for the Cloud', authors:'Mina Sedaghat, Pontus Sköldström, Daniel Turull, Vinay Yadhav, Joacim Halén, Madhubala Ganesan, Amardeep Mehta and Wolfgang John', venue:'arXiv preprint', year:'2020', type:'preprint', url:'https://arxiv.org/abs/2006.07163'},
  {title:'Why Cloud Applications Are Not Ready for the Edge (Yet)', authors:'Chanh Le Tan Nguyen, Amardeep Mehta, Cristian Klein and Erik Elmroth', venue:'ACM/IEEE SEC', year:'2019', url:'https://dl.acm.org/doi/10.1145/3318216.3363298'},
  {title:'Utility-based Allocation of Industrial IoT Applications in Mobile Edge Clouds', authors:'Amardeep Mehta, Ewnetu Bayuh Lakew, Johan Tordsson and Erik Elmroth', venue:'IEEE IPCCC', year:'2018', url:'https://ieeexplore.ieee.org/document/8711075'},
  {title:'Distributed Cost-Optimized Placement for Latency-Critical Applications in Heterogeneous Environments', authors:'Amardeep Mehta and Erik Elmroth', venue:'IEEE ICAC', year:'2018', url:'https://ieeexplore.ieee.org/document/8498133'},
  {title:'Calvin Constrained — A Framework for IoT Applications in Heterogeneous Environments', authors:'Amardeep Mehta et al.', venue:'IEEE ICDCS', year:'2017', url:'https://ieeexplore.ieee.org/document/7980047'},
  {title:'Online Spike Detection in Cloud Workloads', authors:'Amardeep Mehta, Jonas Durango, Johan Tordsson and Erik Elmroth', venue:'IEEE IC2E', year:'2015', url:'https://ieeexplore.ieee.org/document/7092959'},
];
export const thesis = {title:'Resource Allocation for Mobile Edge Clouds', year:'2018', university:'Umeå University', url:'https://umu.diva-portal.org/smash/get/diva2%3A1245153/SPIKBLAD02.pdf'};
