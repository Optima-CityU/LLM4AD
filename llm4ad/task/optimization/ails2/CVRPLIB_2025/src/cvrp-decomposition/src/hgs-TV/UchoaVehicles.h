#ifndef GENVRP_UCHOAVEHICLES_H
#define GENVRP_UCHOAVEHICLES_H

#include <unordered_map>
#include <string>

namespace genvrp {
    static const std::unordered_map<std::string, int> UCHOA_VEHICLES{
        { "X-n101-k25.vrp", 27 },
        { "X-n106-k14.vrp", 15 },
        { "X-n110-k13.vrp", 14 },
        { "X-n115-k10.vrp", 11 },
        { "X-n120-k6.vrp", 7 },
        { "X-n125-k30.vrp", 31 },
        { "X-n129-k18.vrp", 19 },
        { "X-n134-k13.vrp", 14 },
        { "X-n139-k10.vrp", 11 },
        { "X-n143-k7.vrp", 8 },
        { "X-n148-k46.vrp", 48 },
        { "X-n153-k22.vrp", 24 },
        { "X-n157-k13.vrp", 14 },
        { "X-n162-k11.vrp", 12 },
        { "X-n167-k10.vrp", 11 },
        { "X-n172-k51.vrp", 54 },
        { "X-n176-k26.vrp", 27 },
        { "X-n181-k23.vrp", 24 },
        { "X-n186-k15.vrp", 16 },
        { "X-n190-k8.vrp", 9 },
        { "X-n195-k51.vrp", 54 },
        { "X-n200-k36.vrp", 37 },
        { "X-n204-k19.vrp", 20 },
        { "X-n209-k16.vrp", 17 },
        { "X-n214-k11.vrp", 12 },
        { "X-n219-k73.vrp", 74 },
        { "X-n223-k34.vrp", 35 },
        { "X-n228-k23.vrp", 24 },
        { "X-n233-k16.vrp", 18 },
        { "X-n237-k14.vrp", 15 },
        { "X-n242-k48.vrp", 49 },
        { "X-n247-k50.vrp", 52 },
        { "X-n251-k28.vrp", 29 },
        { "X-n256-k16.vrp", 18 },
        { "X-n261-k13.vrp", 14 },
        { "X-n266-k58.vrp", 59 },
        { "X-n270-k35.vrp", 37 },
        { "X-n275-k28.vrp", 29 },
        { "X-n280-k17.vrp", 18 },
        { "X-n284-k15.vrp", 16 },
        { "X-n289-k60.vrp", 62 },
        { "X-n294-k50.vrp", 52 },
        { "X-n298-k31.vrp", 32 },
        { "X-n303-k21.vrp", 22 },
        { "X-n308-k13.vrp", 14 },
        { "X-n313-k71.vrp", 73 },
        { "X-n317-k53.vrp", 54 },
        { "X-n322-k28.vrp", 29 },
        { "X-n327-k20.vrp", 21 },
        { "X-n331-k15.vrp", 16 },
        { "X-n336-k84.vrp", 87 },
        { "X-n344-k43.vrp", 44 },
        { "X-n351-k40.vrp", 42 },
        { "X-n359-k29.vrp", 30 },
        { "X-n367-k17.vrp", 18 },
        { "X-n376-k94.vrp", 95 },
        { "X-n384-k52.vrp", 54 },
        { "X-n393-k38.vrp", 39 },
        { "X-n401-k29.vrp", 30 },
        { "X-n411-k19.vrp", 20 },
        { "X-n420-k130.vrp", 132 },
        { "X-n429-k61.vrp", 63 },
        { "X-n439-k37.vrp", 38 },
        { "X-n449-k29.vrp", 30 },
        { "X-n459-k26.vrp", 27 },
        { "X-n469-k138.vrp", 141 },
        { "X-n480-k70.vrp", 71 },
        { "X-n491-k59.vrp", 61 },
        { "X-n502-k39.vrp", 40 },
        { "X-n513-k21.vrp", 22 },
        { "X-n524-k153.vrp", 158 },
        { "X-n536-k96.vrp", 98 },
        { "X-n548-k50.vrp", 51 },
        { "X-n561-k42.vrp", 43 },
        { "X-n573-k30.vrp", 31 },
        { "X-n586-k159.vrp", 160 },
        { "X-n599-k92.vrp", 95 },
        { "X-n613-k62.vrp", 63 },
        { "X-n627-k43.vrp", 44 },
        { "X-n641-k35.vrp", 36 },
        { "X-n655-k131.vrp", 132 },
        { "X-n670-k130.vrp", 137 },
        { "X-n685-k75.vrp", 76 },
        { "X-n701-k44.vrp", 45 },
        { "X-n716-k35.vrp", 36 },
        { "X-n733-k159.vrp", 161 },
        { "X-n749-k98.vrp", 100 },
        { "X-n766-k71.vrp", 72 },
        { "X-n783-k48.vrp", 49 },
        { "X-n801-k40.vrp", 41 },
        { "X-n819-k171.vrp", 175 },
        { "X-n837-k142.vrp", 143 },
        { "X-n856-k95.vrp", 96 },
        { "X-n876-k59.vrp", 60 },
        { "X-n895-k37.vrp", 39 },
        { "X-n916-k207.vrp", 210 },
        { "X-n936-k151.vrp", 161 },
        { "X-n957-k87.vrp", 88 },
        { "X-n979-k58.vrp", 59 },
        { "X-n1001-k43.vrp", 44 }
    };

    namespace {
        // Substitutes std::filesystem::path::filename for systems
        // without std::filesystem. Not portable! Cave canem!
        std::string filename(const std::string& path) {
            return path.substr(path.find_last_of('/') + 1);
        }
    }

    inline int getUchoaVehicles(std::string pathToInstance) {
        const auto fileName = filename(pathToInstance);
        const auto search = UCHOA_VEHICLES.find(fileName);

        return (search == UCHOA_VEHICLES.end() ? -1 : search->second);
    }
}

#endif //GENVRP_UCHOAVEHICLES_H