from idnns.networks import information_network as inet

def main():
    net = inet.informationNetwork()
    net.print_information()

    print('Start running the network')
    net.run_network()

    print ('Saving data')
    net.save_data()

    print ('Ploting figures')
    net.plot_network()

if __name__ == '__main__':
    main()
