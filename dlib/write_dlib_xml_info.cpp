#include "write_dlib_xml_info.hpp"

#include <pugixml.hpp>

#include <boost/filesystem.hpp>

#include <fstream>

namespace ocv{

namespace odlib{

namespace{

pugi::xml_node write_imglab_xml_preparation(size_t image_location_size,
                                            size_t rect_size,
                                            const std::string &output_name,
                                            std::ofstream &stream, pugi::xml_document &doc)
{
    stream.open(output_name);
    if(!stream.is_open()){
        throw std::runtime_error("cannot open file:" + output_name);
    }

    if(image_location_size != rect_size){
        throw std::runtime_error("image_location.size() != rects.size()");
    }

    auto declarationNode = doc.append_child(pugi::node_declaration);
    declarationNode.append_attribute("version")  = "1.0";
    declarationNode.append_attribute("encoding") = "ISO-8859-1";
    auto xml_style_node = doc.append_child(pugi::xml_node_type::node_pi);
    xml_style_node.set_name("xml-stylesheet");
    xml_style_node.set_value("type=\"text/xsl\" href=\"image_metadata_stylesheet.xsl\"");

    pugi::xml_node node = doc.append_child("dataset");
    node.append_child("name").append_child(pugi::node_pcdata).set_value("imglab dataset");
    node.append_child("comment").append_child(pugi::node_pcdata).set_value("Created by imglab tool.");

    return node.append_child("images");
}

void add_image_info(pugi::xml_node &image, std::vector<dlib::rectangle> const &rects)
{
    for(auto const &rect : rects){
        pugi::xml_node box = image.append_child("box");
        box.append_attribute("top") = rect.top();
        box.append_attribute("left") = rect.left();
        box.append_attribute("width") = rect.width();
        box.append_attribute("height") = rect.height();
    }
}

void add_image_info(pugi::xml_node &image, std::vector<dlib::mmod_rect> const &rects)
{
    for(auto const &mrect : rects){
        pugi::xml_node box = image.append_child("box");
        auto const &rect = mrect.rect;
        box.append_attribute("top") = rect.top();
        box.append_attribute("left") = rect.left();
        box.append_attribute("width") = rect.width();
        box.append_attribute("height") = rect.height();
        if(mrect.ignore){
            box.append_attribute("height") = "1";
        }
    }
}

template<typename T>
void write_imglab_xml_impl(const std::vector<std::string> &image_location, const std::vector<std::vector<T>> &rects,
                           const std::string &prepend_folder, const std::string &output_name)
{
    std::ofstream stream;
    pugi::xml_document doc;
    pugi::xml_node images = write_imglab_xml_preparation(image_location.size(), rects.size(), output_name, stream, doc);

    using namespace boost::filesystem;
    for(size_t i = 0; i != image_location.size(); ++i){
        pugi::xml_node image = images.append_child("image");
        image.append_attribute("file") = (prepend_folder + "/" + path(image_location[i]).filename().string()).c_str();
        add_image_info(image, rects[i]);
    }

    doc.print(stream);
}

}

void write_imglab_xml(const std::vector<std::string> &image_location, const std::vector<std::vector<dlib::rectangle>> &rects,
                      const std::string &prepend_folder, const std::string &output_name)
{
    write_imglab_xml_impl(image_location, rects, prepend_folder, output_name);
}

void write_imglab_xml(const std::vector<std::string> &image_location, const std::vector<std::vector<dlib::mmod_rect>> &rects,
                      const std::string &prepend_folder, const std::string &output_name)
{
    write_imglab_xml_impl(image_location, rects, prepend_folder, output_name);
}

}

}
