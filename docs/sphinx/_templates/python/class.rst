{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

.. py:currentmodule:: {{ obj.id[:-(obj.qual_name|length + 1)] }}

   {% endif %}
   {% set visible_children = obj.children|selectattr("display")|list %}
   {% set own_page_children = (
      visible_children|selectattr("type", "in", own_page_types)|list
   ) %}
   {% if is_own_page and own_page_children %}
.. toctree::
   :hidden:

      {% for child in own_page_children %}
   {{ child.include_path }}
      {% endfor %}

   {% endif %}
   {% set class_name = obj.qual_name if is_own_page else obj.short_name %}
   {% set type_params = "[" ~ obj.type_params ~ "]" if obj.type_params else "" %}
   {% set arguments = "(" ~ obj.args ~ ")" if obj.args else "" %}
.. py:{{ obj.type }}:: {{ class_name }}{{ type_params }}{{ arguments }}

   {% for (args, return_annotation) in obj.overloads %}
      {% set overload_signature = obj.short_name ~ ("(" ~ args ~ ")" if args else "") %}
      {{ " " * (obj.type | length) }}   {{ overload_signature }}

   {% endfor %}
   {% if obj.bases %}
      {% if "show-inheritance" in autoapi_options %}

         {% set bases -%}
            {%- for base in obj.bases -%}
               {{ base|link_objs }}{% if not loop.last %}, {% endif %}
            {%- endfor -%}
         {%- endset %}
   Bases: {{ bases }}
      {% endif %}


      {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
   .. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
      :parts: 1
         {% if "private-members" in autoapi_options %}
      :private-bases:
         {% endif %}

      {% endif %}
   {% endif %}
   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
   {% set groups = namespace(
      attributes=[],
      properties=[],
      class_and_static_methods=[],
      dunder_methods=[],
      methods=[]
   ) %}
   {% for obj_item in visible_children %}
      {% if obj_item.type not in own_page_types %}
         {% if obj_item.type == "attribute" %}
            {% set groups.attributes = groups.attributes + [obj_item] %}
         {% elif obj_item.type == "property" %}
            {% set groups.properties = groups.properties + [obj_item] %}
         {% elif obj_item.type == "method" %}
            {% if obj_item.is_special_member %}
               {% set groups.dunder_methods = groups.dunder_methods + [obj_item] %}
            {% elif (
               "classmethod" in obj_item.properties
               or "staticmethod" in obj_item.properties
            ) %}
               {% set groups.class_and_static_methods = (
                  groups.class_and_static_methods + [obj_item]
               ) %}
            {% else %}
               {% set groups.methods = groups.methods + [obj_item] %}
            {% endif %}
         {% endif %}
      {% endif %}
   {% endfor %}
   {% for title, members in [
      ("Attributes", groups.attributes),
      ("Properties", groups.properties),
      ("Class and static methods", groups.class_and_static_methods),
      ("Dunder methods", groups.dunder_methods),
      ("Methods", groups.methods),
   ] %}
      {% if members %}

   {{ title }}
   {{ "-" * title | length }}

         {% for obj_item in members|sort(attribute="name") %}

   {{ obj_item.render()|indent(3) }}
         {% endfor %}
      {% endif %}
   {% endfor %}
   {% if is_own_page and own_page_children %}
      {% set visible_attributes = (
         own_page_children|selectattr("type", "equalto", "attribute")|list
      ) %}
      {% if visible_attributes %}
Attributes
----------

.. autoapisummary::

         {% for attribute in visible_attributes %}
   {{ attribute.id }}
         {% endfor %}


      {% endif %}
      {% set visible_exceptions = (
         own_page_children|selectattr("type", "equalto", "exception")|list
      ) %}
      {% if visible_exceptions %}
Exceptions
----------

.. autoapisummary::

         {% for exception in visible_exceptions %}
   {{ exception.id }}
         {% endfor %}


      {% endif %}
      {% set visible_classes = (
         own_page_children|selectattr("type", "equalto", "class")|list
      ) %}
      {% if visible_classes %}
Classes
-------

.. autoapisummary::

         {% for klass in visible_classes %}
   {{ klass.id }}
         {% endfor %}


      {% endif %}
      {% set visible_methods = (
         own_page_children|selectattr("type", "equalto", "method")|list
      ) %}
      {% if visible_methods %}
Methods
-------

.. autoapisummary::

            {% for method in visible_methods %}
   {{ method.id }}
            {% endfor %}


      {% endif %}
   {% endif %}
{% endif %}
