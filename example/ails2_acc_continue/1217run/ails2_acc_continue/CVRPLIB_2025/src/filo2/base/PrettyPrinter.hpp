#ifndef _FILO2_PRETTYPRINTER_HPP_
#define _FILO2_PRETTYPRINTER_HPP_

#include <string>
#include <vector>

namespace cobra {

    class PrettyPrinter {

    public:
        class Field {
        public:
            enum Type { INTEGER, REAL, STRING };
            Field(std::string name_, Type type_ = Type::REAL, int max_width_ = 8, std::string sep_ = "\t", int precision_ = 2)
                : name(std::move(name_)), type(type_), max_width(max_width_), sep(std::move(sep_)), precision(precision_){};
            const std::string& get_name() const {
                return name;
            }
            const Type& get_type() const {
                return type;
            }
            int get_max_width() const {
                return max_width;
            }
            const std::string& get_separator() const {
                return sep;
            }
            int get_precision() const {
                return precision;
            }

        private:
            std::string name;
            Type type;
            int max_width;
            std::string sep;
            int precision;
        };

        explicit PrettyPrinter(std::vector<Field> args_) : args(std::move(args_)) { }

        template <typename T, typename... Values>
        void print(T t, Values... values) {  // recursive variadic function

            if (!header_count) {
                header_count = max_header_count;

                std::cout << "\n";

                std::cout << "\033[1m";

                for (auto header : args) {
                    std::cout << " ";
                    std::cout << std::fixed;
                    std::cout << std::setw(header.get_max_width());
                    std::cout << header.get_name();
                    std::cout << std::setprecision(10);
                    std::cout << std::defaultfloat;
                    std::cout << " ";
                    std::cout << header.get_separator();
                }

                std::cout << "\033[0m";

                std::cout << "\n";

                for (auto header : args) {
                    for (auto n = 0; n < header.get_max_width() + 2; n++) {
                        std::cout << " ";
                    }
                    std::cout << header.get_separator();
                }
                std::cout << "\n";
            }
            header_count--;

            if (style != NONE) {
                std::cout << "\033[" + std::to_string(style) + "m";
            }

            print_impl(0, t, values...);

            if (style != NONE) {
                std::cout << "\033[0m";
            }

            std::cout << "\n";
        }

        void notify(const std::string& message) {
            std::cout << "\n";

            if (style != NONE) {
                std::cout << "\033[" + std::to_string(style) + "m";
            }

            std::cout << message << "\n";

            if (style != NONE) {
                std::cout << "\033[0m";
            }

            std::cout << "\n";
        }

        enum Style {
            NONE = 0,
            FOREGROUND_BLACK = 30,
            FOREGROUND_RED,
            FOREGROUND_GREEN,
            FOREGROUND_YELLOW,
            FOREGROUND_BLUE,
            FOREGROUND_MAGENTA,
            FOREGROUND_CYAN,
            FOREGROUND_WHITE,
            BACKGROUND_BLACK = 40,
            BACKGROUND_RED,
            BACKGROUND_GREEN,
            BACKGROUND_YELLOW,
            BACKGROUND_BLUE,
            BACKGROUND_MAGENTA,
            BACKGROUND_CYAN,
            BACKGROUND_WHITE
        };

        void set_style(Style style_) {
            this->style = style_;
        }

        void unset_style() {
            this->style = NONE;
        }

    private:
        template <typename T>
        void just_print(T value, Field& header) {
            std::cout << " ";
            std::cout << std::fixed;
            std::cout << std::setprecision(header.get_precision());
            std::cout << std::setw(header.get_max_width());
            switch (header.get_type()) {
            case Field::INTEGER:
                std::cout << static_cast<long>(value);
                break;
            case Field::REAL:
                std::cout << static_cast<double>(value);
                break;
            case Field::STRING:
                std::cout << value;
                break;
            }
            std::cout << " ";
            std::cout << std::setprecision(10);
            std::cout << std::defaultfloat;
        }

        template <typename T>
        void print_impl(int n, T value) {

            if (static_cast<unsigned>(n) == args.size()) {
                std::cout << "Values do not correspond to headers\n";
                return;
            }

            just_print(value, args[n]);
        }

        template <typename T, typename... Values>
        void print_impl(int n, T value, Values... values)  // recursive variadic function
        {

            if (static_cast<unsigned>(n) == args.size()) {
                std::cout << "Values do not correspond to headers\n";
                return;
            }

            just_print(value, args[n]);
            std::cout << args[n].get_separator();

            print_impl(n + 1, values...);
        }

        std::vector<Field> args;
        int max_header_count = 15;
        int header_count = 0;
        Style style = NONE;
    };

}  // namespace cobra

#endif
