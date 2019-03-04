#include "read_imglab_xml_info.hpp"

#include <algorithm>

#include <dlib/xml_parser.h>

namespace ocv{


namespace odlib{

namespace{

template<typename rect_type = dlib::mmod_rect>
class doc_handler : public dlib::document_handler
{
public:
    doc_handler(std::vector<std::string> &img_name,
                std::vector<std::vector<rect_type>> &location) :
        img_name_(img_name),
        location_(location)
    {

    }

    void start_document () override
    {
    }

    void end_document () override
    {
    }

    void start_element (
            const unsigned long,
            const std::string& name,
            const dlib::attribute_list& atts) override
    {
        if(name == "image"){
            atts.reset();
            if(atts.move_next()){
                img_name_.emplace_back(atts.element().value());
                location_.emplace_back(std::vector<rect_type>());
            }
        }else if(name == "box"){
            atts.reset();
            rect_type rect;
            //dlib::mmod_rect
            long h = 0;
            enum {height, left, top, width, ignore};
            for(size_t index = 0; atts.move_next(); ++index){
                std::string const &attr_name = atts.element().key();
                process_attr(attr_name, atts, rect, h);
            }
            process_bottom(rect, h);
            location_.back().emplace_back(rect);
        }
    }

    void end_element (
            const unsigned long,
            const std::string&
            ) override
    {
    }

    void characters (
            const std::string&
            ) override
    {
    }

    void processing_instruction (const unsigned long, const std::string&,
                                 const std::string&
                                 ) override
    {
    }

private:
    void process_attr(std::string const &name,
                      dlib::attribute_list const &atts,
                      dlib::mmod_rect &rect, long &h) const
    {
        if(name == "ignore"){
            rect.ignore = true;
        }else{
            process_attr(name, atts, rect.rect, h);
        }
    }

    void process_attr(std::string const &name,
                      dlib::attribute_list const &atts,
                      dlib::rectangle &rect, long &h) const
    {
        if(name == "height"){
            h = std::stol(atts.element().value()) - 1;
            h = std::max<long>(0, h);
        }else if(name == "left"){
            rect.set_left(std::stol(atts.element().value()));
        }else if(name == "top"){
            rect.set_top(std::stol(atts.element().value()));
        }else if(name == "width"){
            rect.set_right(rect.left() + std::stol(atts.element().value()) - 1);
            rect.set_right(std::max<long>(0, rect.right()));
        }
    }

    void process_bottom(dlib::mmod_rect &rect, long h) const
    {
        process_bottom(rect.rect, h);
    }

    void process_bottom(dlib::rectangle &rect, long h) const
    {
        rect.set_bottom(rect.top() + h);
    }

    std::vector<std::string> &img_name_;
    std::vector<std::vector<rect_type>> &location_;
};

} //nameless namespace

void read_imglab_xml_info(std::string const &file_name,
                          std::vector<std::string> &img_name,
                          std::vector<std::vector<dlib::rectangle>> &roi)
{
    doc_handler<dlib::rectangle> handler(img_name, roi);
    dlib::parse_xml(file_name, handler);
}

void read_imglab_xml_info(std::string const &file_name,
                          std::vector<std::string> &img_name,
                          std::vector<std::vector<dlib::mmod_rect>> &roi)
{
    doc_handler<dlib::mmod_rect> handler(img_name, roi);
    dlib::parse_xml(file_name, handler);
}

}

}
