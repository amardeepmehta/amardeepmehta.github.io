
export type Patent = {
  title: string;
  inventors: string;
  assignee: string;
  status: 'Granted' | 'Published application';
  primaryNumber: string;
  date: string;
  url: string;
  description: string;
  related: {
    label: string;
    url: string;
  }[];
};

export const patents: Patent[] = [
  {
    title: 'Methods and apparatuses for providing a function as a service platform',
    inventors: 'Vinay Yadhav, Joacim Halén, Amardeep Mehta',
    assignee: 'Ericsson',
    status: 'Granted',
    primaryNumber: 'US 12,210,854 B2',
    date: '28 January 2025',
    url: 'https://patents.google.com/patent/US12210854B2/en',
    description:
      'Intent-based function discovery and invocation using a function catalogue in a Function-as-a-Service platform.',
    related: [
      {
        label: 'US 20220283789 A1',
        url: 'https://patents.google.com/patent/US20220283789A1/en'
      },
      {
        label: 'EP 4018305 B1',
        url: 'https://patents.google.com/patent/EP4018305B1/en'
      },
      {
        label: 'WO 2021040582 A1',
        url: 'https://patents.google.com/patent/WO2021040582A1/en'
      }
    ]
  },
  {
    title: 'Synchronizing conflict-free replicated data types',
    inventors: 'Pontus Sköldström, Amardeep Mehta, Mina Sedaghat, Vinay Yadhav',
    assignee: 'Ericsson',
    status: 'Granted',
    primaryNumber: 'US 12,254,025 B2',
    date: '18 March 2025',
    url: 'https://patents.google.com/patent/US12254025B2/en',
    description:
      'Synchronization of conflict-free replicated data types across distributed devices using operation-based and state-based synchronization.',
    related: [
      {
        label: 'US 20230297588 A1',
        url: 'https://patents.google.com/patent/US20230297588A1/en'
      },
      {
        label: 'EP 4179432 A1',
        url: 'https://patents.google.com/patent/EP4179432A1/en'
      },
      {
        label: 'WO 2022010388 A1',
        url: 'https://patents.google.com/patent/WO2022010388A1/en'
      }
    ]
  },
  {
    title: 'Methods, host devices and computer program products for assigning communication ports',
    inventors: 'Amardeep Mehta, Pontus Sköldström, Vinay Yadhav',
    assignee: 'Ericsson',
    status: 'Published application',
    primaryNumber: 'WO 2021188026 A1',
    date: '23 September 2021',
    url: 'https://patents.google.com/patent/WO2021188026A1/en',
    description:
      'Dynamic allocation of communication addresses to resolve port conflicts among applications sharing a host.',
    related: []
  },
  {
    title: 'Availability handling in a wireless communication system',
    inventors: 'Lackis Eleftheriadis, Amardeep Mehta',
    assignee: 'Ericsson',
    status: 'Published application',
    primaryNumber: 'WO 2023191683 A1',
    date: '5 October 2023',
    url: 'https://patents.google.com/patent/WO2023191683A1/en',
    description:
      'Dynamic management of power supply and compute availability to satisfy service requirements in wireless edge infrastructure.',
    related: []
  }
];
