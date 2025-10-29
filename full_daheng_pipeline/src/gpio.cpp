#include "../include/gpio.hpp"

GPIO::GPIO(int num) {
    std::string path = "/sys/class/gpio/export";
    int fd = open(path.c_str(), O_RDWR | O_SYNC);
    write(gpio, &num, sizeof(int));
    close(fd);

    fd = open( std::string( "/sys/class/gpio/gpio").append(std::to_string(num)).append("/direction").c_str(), O_RDWR | O_SYNC ); 
    
    write(fd, "out", 4 * sizeof(char));

    close(fd);

    gpio = open( std::string( "/sys/class/gpio/gpio").append(std::to_string(num)).append("/value").c_str(), O_RDWR | O_SYNC ); 
}

void GPIO::init(int num) {

    number = num;
    std::string path = "/sys/class/gpio/export";
    int fd = open(path.c_str(), O_RDWR | O_SYNC);
    write(gpio, &num, sizeof(int));
    close(fd);

    fd = open( std::string( "/sys/class/gpio/gpio").append(std::to_string(num)).append("/direction").c_str(), O_RDWR | O_SYNC ); 
    
    write(fd, "out", 4 * sizeof(char));

    close(fd);

    

}

void GPIO::send(std::string value) {
    std::cout << "STRING : " << std::string( "/sys/class/gpio/gpio").append(std::to_string(number)).append("/value") << std::endl;
    gpio = open( std::string( "/sys/class/gpio/gpio").append(std::to_string(number)).append("/value").c_str(), O_RDWR | O_SYNC ); 
    int status = write(gpio, value.c_str(), sizeof(int));
    std::cout << "Status "<< status << std::endl;
    close(gpio);
}

void GPIO::get(int* value) {
    read(gpio, value, sizeof(int));
}

void GPIO::trigger() {
    send("0");
    usleep(1000 * 16);
    send("1");
}

