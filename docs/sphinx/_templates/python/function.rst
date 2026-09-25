{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

.. py:currentmodule:: {{ obj.id[:-(obj.qual_name|length + 1)] }}

   {% endif %}
   {% set type_params = "[" ~ obj.type_params ~ "]" if obj.type_params else "" %}
   {% set return_annotation = (
      " -> " ~ obj.return_annotation
      if obj.return_annotation is not none
      else ""
   ) %}
   {% set signature = (
      obj.short_name ~ type_params ~ "(" ~ obj.args ~ ")" ~ return_annotation
   ) %}
.. py:function:: {{ signature }}
   {% for (args, return_annotation) in obj.overloads %}

      {% set overload_signature = (
         obj.short_name ~ "(" ~ args ~ ")"
         ~ (" -> " ~ return_annotation if return_annotation is not none else "")
      ) %}
                 {{ overload_signature }}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
