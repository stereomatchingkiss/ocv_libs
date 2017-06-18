#include "write_dlib_xml_info.hpp"

#include <pugixml.hpp>

#include <boost/filesystem.hpp>

#include <fstream>

namespace ocv{

namespace odlib{

void write_dlib_xml(const std::vector<std::string> &image_location, const std::vector<std::vector<dlib::rectangle>> &rects,
                    const std::string &prepend_folder, const std::string &output_name)
{
    std::ofstream stream(output_name);
    if(!stream.is_open()){
        throw std::runtime_error("cannot open file:" + output_name);
    }

    if(image_location.size() != rects.size()){
        throw std::runtime_error("image_location.size() != rects.size()");
    }

    pugi::xml_document doc;
    pugi::xml_node node = doc.append_child("dataset");
    node.append_child("name").append_child(pugi::node_pcdata).set_value("imglab dataset");
    node.append_child("comment").append_child(pugi::node_pcdata).set_value("Created by imglab tool.");
    pugi::xml_node images = node.append_child("images");

    using namespace boost::filesystem;

    for(size_t i = 0; i != image_location.size(); ++i){
        pugi::xml_node image = images.append_child("image");
        image.append_attribute("file") = (prepend_folder + "/" + path(image_location[i]).filename().string()).c_str();
        for(auto const &rect : rects[i]){
            pugi::xml_node box = image.append_child("box");
            box.append_attribute("top") = rect.top();
            box.append_attribute("left") = rect.left();
            box.append_attribute("width") = rect.width();
            box.append_attribute("height") = rect.height();
        }
    }

    doc.print(stream);
}

void write_dlib_xml(const std::vector<std::string> &image_location, const std::vector<std::vector<dlib::mmod_rect>> &rects,
                    const std::string &prepend_folder, const std::string &output_name)
{
    std::vector<std::vector<dlib::rectangle>> simple_rects;
    simple_rects.reserve(rects.size());
    for(auto const &rect : rects){
        std::vector<dlib::rectangle> temp;
        for(auto const &r : rect){
            temp.emplace_back(r.rect);
        }
        simple_rects.emplace_back(std::move(temp));
    }

    write_dlib_xml(image_location, simple_rects, prepend_folder, output_name);
}

}

}
